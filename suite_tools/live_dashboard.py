"""Serve a local live dashboard for benchmark run ledgers.

Benchmark CLIs remain the source of truth for execution; this module watches
``RUN_STATUS.json`` and ``RUN_EVENTS.jsonl`` files and presents them in a
browser. The only dashboard write is the non-destructive ``RUN_DISPOSITION``
sidecar used to exclude malformed diagnostic runs from analysis views without
editing run ledgers, transcripts, scores, or contracts.
"""

from __future__ import annotations

import argparse
import copy
import getpass
import hashlib
import html
import ipaddress
import json
import os
import secrets
import shlex
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from suite_tools.dashboard_store import DashboardStore
from suite_tools.env import read_repo_env_values
from suite_tools.model_config import load_suite_config, validate_suite_config
from suite_tools.paid_call_lease import (
    LEASE_STATUS_FILENAME,
    PAID_CALL_LIMIT_ENV_NAMES,
    POLICY_FILENAME,
    configured_max_active_calls,
    default_lease_dir,
    load_paid_call_lease_status,
    paid_call_capacity_report,
)
from suite_tools.run_contract import (
    CONTRACT_FILENAME,
    CONTROL_FILENAME,
    PLAN_FILENAME,
    load_run_contract,
    load_run_control,
    load_run_plan,
    summarize_contract,
    summarize_control,
)
from suite_tools.run_monitor import sanitize_error_message, sanitize_ledger_value
from suite_tools.run_monitor import atomic_write_json, utc_now
from suite_tools.progress_dedupe import (
    completed_unit_keys,
    event_unit_key as _dedupe_event_unit_key,
    COMPLETED_EVENTS as _DEDUPE_COMPLETED_EVENTS,
    REUSED_EVENTS as _DEDUPE_REUSED_EVENTS,
    TERMINAL_SIGNAL_EVENTS as _DEDUPE_TERMINAL_SIGNAL_EVENTS,
    SCORING_COMPLETED_EVENTS as _DEDUPE_SCORING_COMPLETED_EVENTS,
    ALL_PROGRESS_EVENTS as _DEDUPE_ALL_PROGRESS_EVENTS,
)
from suite_tools.scheduler import (
    SCHEDULER_STATUS_FILENAME,
    load_scheduler_status,
)


def _scheduler_run_command(
    contract_path: Any,
    max_active_calls: Any = None,
) -> str:
    display_path = _display_path(contract_path)
    if not display_path:
        return ""
    argv = [
        "./venv/bin/python",
        "-m",
        "suite_tools.scheduler",
        "run",
        "--contract",
        display_path,
    ]
    limit = max_active_calls or configured_max_active_calls()
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        argv.extend(["--max-active-calls", str(limit)])
    argv.append("--stop-on-attention")
    return shlex.join(argv)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "prepared"
DEFAULT_SUITE_CONFIG = REPO_ROOT / "suite_models.yaml"
DASHBOARD_ASSETS_DIR = Path(__file__).resolve().parent / "dashboard_assets"
BRAND_LOGO_STATIC_NAME = "anti-sycophancy-logo-static.png"
BRAND_LOGO_RUNNING_NAME = "anti-sycophancy-logo-running.gif"
DASHBOARD_ASSETS = {
    f"/assets/{BRAND_LOGO_STATIC_NAME}": (
        DASHBOARD_ASSETS_DIR / BRAND_LOGO_STATIC_NAME,
        "image/png",
    ),
    f"/assets/{BRAND_LOGO_RUNNING_NAME}": (
        DASHBOARD_ASSETS_DIR / BRAND_LOGO_RUNNING_NAME,
        "image/gif",
    ),
    "/assets/dashboard.css": (
        DASHBOARD_ASSETS_DIR / "dashboard.css",
        "text/css; charset=utf-8",
    ),
    "/assets/dashboard.js": (
        DASHBOARD_ASSETS_DIR / "dashboard.js",
        "text/javascript; charset=utf-8",
    ),
    "/assets/theme-init.js": (
        DASHBOARD_ASSETS_DIR / "theme-init.js",
        "text/javascript; charset=utf-8",
    ),
    "/favicon.ico": (
        DASHBOARD_ASSETS_DIR / BRAND_LOGO_STATIC_NAME,
        "image/png",
    ),
}
# Concurrent pollers and detail requests share the same revision-bound build.
BUILD_WAIT_TIMEOUT_SECONDS = 30.0
STALE_RUNNING_SECONDS = 15 * 60
# Staleness and elapsed-time labels change even when no ledger file is written.
# Keep conditional polling cheap while forcing those watchdogs to reevaluate.
DASHBOARD_WATCHDOG_REVISION_SECONDS = 30
DISPOSITION_FILENAME = "RUN_DISPOSITION.json"
DISPOSITION_EVENTS_FILENAME = "RUN_DISPOSITION_EVENTS.jsonl"
DISPOSITION_SCHEMA_VERSION = "benchmark-run-disposition-v1"
MAX_MODULE_EVIDENCE_ITEMS = 260
MAX_TRANSCRIPT_EVIDENCE_TURNS = 20
MAX_LIVE_EVIDENCE_ITEMS = 1000
RESULTS_LIFECYCLE_DIRS = ("prepared", "testing", "final", "internal")

_CONTRACT_CACHE_LOCK = threading.Lock()
_CONTRACT_SUMMARY_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
_CONTRACT_HEADER_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
_SUITE_INVENTORY_CACHE: dict[str, tuple[tuple[int, int] | None, dict[str, Any]]] = {}
_SUITE_INVENTORY_CACHE_LOCK = threading.Lock()
_EVENT_CACHE_LOCK = threading.Lock()
_DISPOSITION_WRITE_LOCK = threading.Lock()
_EVENT_CACHE: dict[tuple[str, str], tuple[tuple[int, int], list[dict[str, Any]]]] = {}
_EVIDENCE_CACHE_LOCK = threading.Lock()
_EVIDENCE_ARTIFACT_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
_MODULE_EVIDENCE_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
_TRANSCRIPT_PREVIEW_CACHE: dict[tuple[Any, ...], dict[str, Any] | None] = {}
MAX_DETAIL_WINDOW = MAX_LIVE_EVIDENCE_ITEMS


def _asset_url(path: str) -> str:
    asset = DASHBOARD_ASSETS.get(path)
    if not asset:
        return ""
    asset_path, _content_type = asset
    return path if asset_path.exists() else ""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


ARCHIVE_DIR_PREFIX = "_archive"


def _path_has_archived_segment(path: Path, root: Path) -> bool:
    """True when any path segment under ``root`` starts with ``_archive``.

    Retired collections are parked in ``_archive_*`` subdirectories (e.g.
    ``_archive_invalid_endpoint_*`` / ``_archive_64k_cap_*``). Their stale
    ledgers must never surface as live run activity.
    """
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        try:
            parts = path.resolve().relative_to(root.resolve()).parts
        except (OSError, ValueError):
            parts = path.parts
    return any(part.startswith(ARCHIVE_DIR_PREFIX) for part in parts)


def _iter_ledger_paths(root: Path, pattern: str) -> list[Path]:
    """Sorted ``rglob`` matches with ``_archive*`` subtrees excluded."""
    return sorted(
        path
        for path in root.rglob(pattern)
        if not _path_has_archived_segment(path, root)
    )


def _load_disposition(output_dir: Path) -> dict[str, Any]:
    disposition = sanitize_ledger_value(_load_json(output_dir / DISPOSITION_FILENAME))
    if not isinstance(disposition, dict):
        return {}
    if disposition.get("schema_version") != DISPOSITION_SCHEMA_VERSION:
        return {}
    return disposition


def _disposition_eligibility(status: dict[str, Any], disposition: dict[str, Any]) -> dict[str, bool]:
    """Derive dashboard eligibility from the immutable run ledger, not a sidecar."""
    if _is_rejected_from_analysis(disposition):
        return {
            "eligible_for_generation": False,
            "eligible_for_scoring": False,
            "eligible_for_promotion": False,
        }
    generation_complete = status.get("status") == "completed"
    score_ready = generation_complete and status.get("validity") == "score_ready"
    return {
        "eligible_for_generation": generation_complete,
        "eligible_for_scoring": score_ready,
        "eligible_for_promotion": score_ready,
    }


def _status_allows_analysis_rejection(status: dict[str, Any]) -> bool:
    """Only failed, nonpublishable ledgers may be excluded from analysis."""
    status_name = status.get("status")
    return (
        isinstance(status_name, str)
        and status_name.startswith("failed_")
        and status.get("validity") != "score_ready"
    )


def _append_disposition_event(path: Path, event: dict[str, Any]) -> None:
    """Persist the audit event before replacing the compatibility snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(event, sort_keys=True, default=str) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _persist_disposition_update(
    event_path: Path,
    event: dict[str, Any],
    snapshot_path: Path,
    snapshot: dict[str, Any],
) -> None:
    """Serialize the durable event and its compatibility snapshot."""
    with _DISPOSITION_WRITE_LOCK:
        _append_disposition_event(event_path, event)
        atomic_write_json(snapshot_path, snapshot)


def _default_operator_id() -> str:
    try:
        user = getpass.getuser().strip()
    except (OSError, KeyError):
        user = "unknown"
    return f"local:{user or 'unknown'}"


def _require_loopback_address(value: str, *, field: str) -> None:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a loopback IP address, not {value!r}") from exc
    if not address.is_loopback:
        raise ValueError(f"{field} must be a loopback IP address, not {value!r}")


def _is_rejected_from_analysis(disposition: dict[str, Any]) -> bool:
    return disposition.get("disposition") == "rejected_from_analysis"


def _tail_lines(path: Path, limit: int) -> list[str]:
    if limit <= 0:
        return []
    block_size = 8192
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            if position <= 0:
                return []
            handle.seek(position - 1)
            needs_extra_newline = handle.read(1) == b"\n"
            needed_newlines = limit + 1 if needs_extra_newline else limit
            chunks: list[bytes] = []
            newline_count = 0
            while position > 0 and newline_count < needed_newlines:
                read_size = min(block_size, position)
                position -= read_size
                handle.seek(position)
                chunk = handle.read(read_size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
    except OSError:
        return []
    data = b"".join(reversed(chunks))
    return [
        line.decode("utf-8", errors="replace")
        for line in data.splitlines()[-limit:]
    ]


def _load_events(path: Path, *, limit: int | None = 80) -> list[dict[str, Any]]:
    try:
        stat = path.stat()
    except OSError:
        return []
    signature = (stat.st_mtime_ns, stat.st_size)
    cache_key = (str(path.resolve()), f"tail:{limit}")
    with _EVENT_CACHE_LOCK:
        cached = _EVENT_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1]
    if limit is not None:
        lines = _tail_lines(path, limit)
    else:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return []
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(sanitize_ledger_value(event))
    with _EVENT_CACHE_LOCK:
        _EVENT_CACHE[cache_key] = (signature, events)
    return events


def _load_events_filtered(path: Path, names: set[str]) -> list[dict[str, Any]]:
    if not names:
        return []
    try:
        stat = path.stat()
    except OSError:
        return []
    signature = (stat.st_mtime_ns, stat.st_size)
    cache_key = (str(path.resolve()), f"names:{','.join(sorted(names))}")
    with _EVENT_CACHE_LOCK:
        cached = _EVENT_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1]
    name_tokens = tuple(name.encode("utf-8") for name in sorted(names) if name)
    events: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                if not any(token in line for token in name_tokens):
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("event") in names:
                    events.append(sanitize_ledger_value(event))
    except OSError:
        return []
    with _EVENT_CACHE_LOCK:
        _EVENT_CACHE[cache_key] = (signature, events)
    return events


def _runs_cache_key(results_root: Path) -> str:
    return str(results_root.resolve())


def _dashboard_source_revision(results_root: Path) -> str:
    """Return a cheap fingerprint for files that can change dashboard data."""
    digest = hashlib.blake2b(digest_size=16)
    watchdog_bucket = int(time.time() // DASHBOARD_WATCHDOG_REVISION_SECONDS)
    digest.update(f"watchdog:{watchdog_bucket}\0".encode())

    def add_path(path: Path, label: str) -> None:
        try:
            stat = path.stat()
        except OSError:
            return
        digest.update(label.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(b":")
        digest.update(str(stat.st_size).encode())
        digest.update(b"\0")

    root = results_root.resolve()
    if root.exists():
        for directory, dirnames, filenames in os.walk(root):
            dirnames.sort()
            # Prune archived subtrees so retired collections never move the
            # source-revision fingerprint (and never surface as live runs).
            dirnames[:] = [
                name for name in dirnames if not name.startswith(ARCHIVE_DIR_PREFIX)
            ]
            directory_path = Path(directory)
            try:
                directory_label = directory_path.relative_to(root).as_posix()
            except ValueError:
                directory_label = directory_path.as_posix()
            # Directory mtimes catch immutable transcript/score creation and removal
            # without stat-ing thousands of completed artifacts on every poll.
            add_path(directory_path, f"directory:{directory_label}")
            for filename in sorted(filenames):
                if not (
                    filename.startswith(("RUN_", "SCHEDULER_"))
                    or filename.endswith("STATUS.json")
                    or filename.endswith("EVENTS.jsonl")
                ):
                    continue
                path = directory_path / filename
                try:
                    label = path.relative_to(root).as_posix()
                except ValueError:
                    label = path.as_posix()
                add_path(path, f"results:{label}")

    add_path(DEFAULT_SUITE_CONFIG, "suite-config")
    lease_dir = default_lease_dir()
    for filename in (LEASE_STATUS_FILENAME, POLICY_FILENAME):
        add_path(lease_dir / filename, f"leases:{filename}")
    return digest.hexdigest()


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _elapsed_seconds(status: dict[str, Any], now: datetime) -> int | None:
    started = _parse_time(status.get("started_at"))
    if started is None:
        return None
    finished = (
        _parse_time(status.get("completed_at"))
        or _parse_time(status.get("failed_at"))
        or now
    )
    return int((finished - started).total_seconds())


def _age_seconds(value: Any, now: datetime) -> int | None:
    timestamp = _parse_time(value)
    if timestamp is None:
        return None
    return max(0, int((now - timestamp).total_seconds()))


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        parts = path.parts
        if "results" in parts:
            return Path(*parts[parts.index("results") :]).as_posix()
        return path.name if path.is_absolute() else path.as_posix()


def _display_path(value: Any, root: Path = REPO_ROOT) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return text
    path = Path(text)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        pass
    parts = path.parts
    if "results" in parts:
        return Path(*parts[parts.index("results") :]).as_posix()
    return path.name


def _display_paths_in_value(value: Any) -> Any:
    if isinstance(value, dict):
        path_keys = {
            "path",
            "contract_path",
            "events_path",
            "expected_transcript_path",
            "output_dir",
            "results_root",
            "score_path",
            "state_dir",
            "status_path",
            "transcript_path",
        }
        return {
            key: _display_path(item) if key in path_keys else _display_paths_in_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_display_paths_in_value(item) for item in value]
    return value


def _cost_summary(status: dict[str, Any]) -> dict[str, Any] | None:
    cost = status.get("cost")
    if not isinstance(cost, dict):
        return None

    def number(value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    total = number(cost.get("total_cost_usd"))
    if total is None:
        return None
    calls = number(cost.get("total_calls"))
    tokens_in = number(cost.get("tokens_in")) or 0
    tokens_out = number(cost.get("tokens_out")) or 0
    thinking_tokens_out = number(cost.get("thinking_tokens_out")) or 0
    billable_tokens_out = number(cost.get("billable_tokens_out"))
    if billable_tokens_out is None:
        billable_tokens_out = tokens_out + thinking_tokens_out
    credit_remaining = number(cost.get("credit_remaining_usd"))
    reported = number(cost.get("reported_cost_usd"))
    estimated = number(cost.get("estimated_cost_usd"))
    classified_total = (reported or 0) + (estimated or 0)
    unclassified = max(0.0, total - classified_total)
    summary = {
        "total_cost_usd": round(total, 4),
        "reported_cost_usd": round(reported or 0, 4),
        "estimated_cost_usd": round(estimated or 0, 4),
        "unclassified_cost_usd": round(unclassified, 4),
        "total_calls": int(calls) if calls is not None else None,
        "tokens": int(tokens_in + tokens_out),
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "thinking_tokens_out": int(thinking_tokens_out),
        "billable_tokens_out": int(billable_tokens_out),
        "billable_tokens": int(tokens_in + billable_tokens_out),
        "credit_remaining_usd": credit_remaining,
    }
    unknown_calls = number(cost.get("unknown_cost_calls"))
    if unknown_calls is not None:
        summary["unknown_cost_calls"] = int(unknown_calls)
    unknown_models = cost.get("unknown_cost_by_model")
    if isinstance(unknown_models, dict) and unknown_models:
        summary["unknown_cost_by_model"] = {
            str(model): int(number(count) or 0)
            for model, count in unknown_models.items()
            if number(count) is not None
        }
    return summary


def _adapter_model_ids(contract: dict[str, Any] | None) -> set[str]:
    model_ids: set[str] = set()
    for model in (contract or {}).get("expected_models") or []:
        if not isinstance(model, dict):
            continue
        metadata = model.get("condition_metadata")
        adapter_profile = (
            metadata.get("adapter_profile")
            if isinstance(metadata, dict)
            else model.get("adapter_profile")
        )
        endpoint = str(model.get("endpoint") or "")
        if not adapter_profile and not endpoint.endswith("_adapter"):
            continue
        for key in ("key", "model_id", "condition_id", "provider_condition_id"):
            value = model.get(key)
            if isinstance(value, str) and value:
                model_ids.add(value)
    return model_ids


def _adapter_only_unpriced_calls(
    status: dict[str, Any],
    contract: dict[str, Any] | None,
) -> bool:
    adapter_models = _adapter_model_ids(contract)
    if not adapter_models:
        return False
    cost = status.get("cost") if isinstance(status.get("cost"), dict) else {}
    usage_by_model = cost.get("usage_by_model")
    if isinstance(usage_by_model, dict):
        unpriced_models = {
            str(model)
            for model, usage in usage_by_model.items()
            if isinstance(usage, dict) and (usage.get("unknown_cost_calls") or 0)
        }
        if unpriced_models:
            return unpriced_models.issubset(adapter_models)
    usage_by_role = cost.get("usage_by_role")
    if not isinstance(usage_by_role, dict):
        return False
    model_unknown = (
        (usage_by_role.get("model_under_test") or {}).get("unknown_cost_calls") or 0
        if isinstance(usage_by_role.get("model_under_test"), dict)
        else 0
    )
    return bool(model_unknown) and model_unknown == (cost.get("unknown_cost_calls") or 0)


def _spend_guard(
    status: dict[str, Any],
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cost = _cost_summary(status)
    if cost is None:
        return {
            "severity": "idle",
            "label": "not tracked",
            "detail": "No cost data has been written for this module.",
        }

    metadata = status.get("metadata") if isinstance(status.get("metadata"), dict) else {}
    raw_budget = status.get("budget_usd") or metadata.get("budget_usd")
    total = cost.get("total_cost_usd") or 0
    credit_remaining = cost.get("credit_remaining_usd")
    unknown_cost_calls = cost.get("unknown_cost_calls") or 0
    if unknown_cost_calls:
        if _adapter_only_unpriced_calls(status, contract):
            return {
                "severity": "info",
                "kind": "adapter_pricing_partial",
                "label": "adapter pricing partial",
                "detail": (
                    f"{unknown_cost_calls} adapter-backed model call(s) did not expose an individual price. "
                    "Displayed spend is partial; this does not affect benchmark evidence or scoring."
                ),
            }
        models = ", ".join(sorted((cost.get("unknown_cost_by_model") or {}).keys()))
        model_note = f" for {models}" if models else ""
        return {
            "severity": "attention",
            "label": "unpriced calls",
            "detail": (
                f"{unknown_cost_calls} paid call(s){model_note} have no cost estimate. "
                "Tracked spend is undercounted. Stop before more paid calls."
            ),
        }
    if isinstance(raw_budget, (int, float)) and total > float(raw_budget):
        return {
            "severity": "attention",
            "label": "budget exceeded",
            "detail": f"Tracked cost ${total:.4f} is above budget ${float(raw_budget):.4f}.",
        }
    if isinstance(credit_remaining, (int, float)) and credit_remaining <= 1:
        return {
            "severity": "attention",
            "label": "low credit",
            "detail": f"Provider credit remaining is ${credit_remaining:.2f}. Stop before more paid calls.",
        }
    if isinstance(credit_remaining, (int, float)) and credit_remaining <= 5:
        return {
            "severity": "warning",
            "label": "watch credit",
            "detail": f"Provider credit remaining is ${credit_remaining:.2f}.",
        }
    return {
        "severity": "ready",
        "label": "within tracked spend",
        "detail": f"Tracked cost is ${total:.4f}.",
    }


def _operator_commands() -> list[dict[str, str]]:
    return [
        {
            "title": "Validate and list config",
            "description": "Offline, free. Confirms central models, groups, and judge sets are parseable.",
            "command": (
                "./venv/bin/python -m suite_tools.model_config --validate\n"
                "./venv/bin/python -m suite_tools.model_config --list\n"
                "./venv/bin/python -m suite_tools.offline_gate"
            ),
        },
        {
            "title": "OpenRouter model/pricing preflight",
            "description": "No generations. Checks OpenRouter-routed slugs and catalog pricing; local/private endpoint ids are skipped.",
            "command": "./venv/bin/python -m suite_tools.openrouter_preflight --config suite_models.yaml",
        },
        {
            "title": "Render tiny-smoke configs",
            "description": "Offline, free. Change group:calibration_smoke or --judge-set calibration to test another selection.",
            "command": (
                "./venv/bin/python -m suite_tools.model_config \\\n"
                "  --judge-set calibration \\\n"
                "  --models group:calibration_smoke \\\n"
                "  --output-dir /tmp/benchmark-configs"
            ),
        },
        {
            "title": "AITA one-item smoke",
            "description": "Paid model-under-test plus judge calls. Runs generation, then scores only if generation completed cleanly.",
            "command": (
                "RUN_ID=operator-smoke-$(date +%Y%m%d-%H%M%S)\n"
                "# Run from the benchmark repo root.\n"
                "cd aita-bench\n"
                "../venv/bin/python -m aita_bench run \\\n"
                "  --config /tmp/benchmark-configs/calibration/aita-models.yaml \\\n"
                "  --models all \\\n"
                "  --items 1 \\\n"
                "  --dataset-mode nta-paired \\\n"
                "  --output ../results/testing/$RUN_ID/aita\n"
                "../venv/bin/python -m aita_bench score \\\n"
                "  --input ../results/testing/$RUN_ID/aita \\\n"
                "  --config /tmp/benchmark-configs/calibration/aita-models.yaml"
            ),
        },
        {
            "title": "Epis one-item smoke",
            "description": "Paid model-under-test plus judge calls. Exercises epistemic sycophancy generation and scoring.",
            "command": (
                "RUN_ID=operator-smoke-$(date +%Y%m%d-%H%M%S)\n"
                "# Run from the benchmark repo root.\n"
                "cd epistemic-sycophancy-bench\n"
                "../venv/bin/python -m epis_bench run \\\n"
                "  --config /tmp/benchmark-configs/calibration/epis-models.yaml \\\n"
                "  --models all \\\n"
                "  --items 1 \\\n"
                "  --output ../results/testing/$RUN_ID/epis\n"
                "../venv/bin/python -m epis_bench score \\\n"
                "  --input ../results/testing/$RUN_ID/epis \\\n"
                "  --config /tmp/benchmark-configs/calibration/epis-models.yaml"
            ),
        },
        {
            "title": "SUS one-scenario smoke",
            "description": "Paid. SUS generation completes first; run scoring only after transcripts are clean.",
            "command": (
                "RUN_ID=operator-smoke-$(date +%Y%m%d-%H%M%S)\n"
                "# Run from the benchmark repo root.\n"
                "cd sus-bench\n"
                "../venv/bin/python -m sus_bench run \\\n"
                "  --models /tmp/benchmark-configs/calibration/sus-models.yaml \\\n"
                "  --runs 1 \\\n"
                "  --scenarios bridge_heights \\\n"
                "  --output ../results/testing/$RUN_ID/sus\n"
                "../venv/bin/python -m sus_bench score \\\n"
                "  --input ../results/testing/$RUN_ID/sus \\\n"
                "  --models /tmp/benchmark-configs/calibration/sus-models.yaml"
            ),
        },
        {
            "title": "Build review viewer",
            "description": "Offline, free. Use after a smoke to inspect transcripts and score artifacts together.",
            "command": (
                "./venv/bin/python -m suite_tools.review_viewer \\\n"
                "  results/testing/$RUN_ID \\\n"
                "  --output /tmp/$RUN_ID-review.html"
            ),
        },
    ]


def _suite_inventory() -> dict[str, Any]:
    cache_key = str(DEFAULT_SUITE_CONFIG.resolve())
    try:
        config_stat = DEFAULT_SUITE_CONFIG.stat()
        signature: tuple[int, int] | None = (config_stat.st_mtime_ns, config_stat.st_size)
    except OSError:
        signature = None
    with _SUITE_INVENTORY_CACHE_LOCK:
        cached = _SUITE_INVENTORY_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return copy.deepcopy(cached[1])

    try:
        config = load_suite_config(DEFAULT_SUITE_CONFIG)
        warnings = validate_suite_config(config)
    except Exception as exc:  # pragma: no cover - defensive browser payload.
        inventory = {
            "config_path": _relative(DEFAULT_SUITE_CONFIG, REPO_ROOT),
            "error": str(exc),
            "warnings": [],
            "judge_sets": [],
            "model_groups": [],
            "models": [],
            "commands": _operator_commands(),
        }
        with _SUITE_INVENTORY_CACHE_LOCK:
            _SUITE_INVENTORY_CACHE[cache_key] = (signature, copy.deepcopy(inventory))
        return inventory

    model_groups = []
    model_to_groups: dict[str, list[str]] = {}
    for name, keys in sorted((config.get("model_groups") or {}).items()):
        keys = list(keys or [])
        model_groups.append({"name": name, "models": keys, "count": len(keys)})
        for key in keys:
            model_to_groups.setdefault(key, []).append(name)

    models = []
    defaults = config.get("defaults") or {}
    for key, model in sorted((config.get("models") or {}).items()):
        endpoint = model.get("endpoint", defaults.get("endpoint", "openrouter"))
        models.append(
            {
                "key": key,
                "label": model.get("label") or key,
                "model_id": model.get("model_id"),
                "endpoint": endpoint,
                "max_parallel": model.get("max_parallel", defaults.get("max_parallel")),
                "groups": model_to_groups.get(key, []),
            }
        )

    judge_sets = []
    for name, judge in sorted((config.get("judge_sets") or {}).items()):
        panel = judge.get("panel") or [judge.get("primary")]
        judge_sets.append(
            {
                "name": name,
                "description": judge.get("description", ""),
                "primary": judge.get("primary"),
                "panel": [item for item in panel if item],
            }
        )

    inventory = {
        "config_path": _relative(DEFAULT_SUITE_CONFIG, REPO_ROOT),
        "schema_version": config.get("schema_version"),
        "warnings": warnings,
        "judge_sets": judge_sets,
        "model_groups": model_groups,
        "models": models,
        "commands": _operator_commands(),
    }
    with _SUITE_INVENTORY_CACHE_LOCK:
        _SUITE_INVENTORY_CACHE[cache_key] = (signature, copy.deepcopy(inventory))
    return inventory


def _shorten(value: Any, *, limit: int = 220) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1].rstrip()}..."


def _severity(status: Any, validity: Any) -> str:
    status_text = str(status or "")
    validity_text = str(validity or "")
    if status_text.startswith("failed"):
        return "attention"
    if status_text in {"running", "starting", "started"}:
        return "running"
    if validity_text == "score_ready":
        return "ready"
    if status_text == "completed" and validity_text == "not_score_ready":
        return "idle"
    if validity_text == "not_score_ready":
        return "attention"
    return "idle"


def _looks_rate_limited(text: Any) -> bool:
    lowered = str(text or "").lower()
    return "429" in lowered or "rate limit" in lowered or "too-many-requests" in lowered


def _looks_credit_exhausted(text: Any) -> bool:
    lowered = str(text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "insufficient_quota",
            "insufficient quota",
            "insufficient credits",
            "credit balance",
            "credits exhausted",
            "quota exhausted",
            "billing quota",
            "http 402",
            "error code: 402",
        )
    )


def _looks_generic_model_error(text: Any) -> bool:
    lowered = str(text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "encountered an error processing",
            "please try again",
            "model failed",
            "provider failed",
            "failed to generate",
            "rate limit",
            "too many requests",
            "429",
        )
    )


def _failure_classification(
    events: list[dict[str, Any]],
    failure_text: Any,
) -> dict[str, str] | None:
    classified = next(
        (
            event
            for event in reversed(events)
            if event.get("event") == "attempt_failure_classified"
        ),
        None,
    )
    if classified is not None:
        fields = (
            "evidence_class",
            "category",
            "action",
            "provider",
            "provider_code",
            "retry_policy_kind",
        )
        result = {
            field: str(classified.get(field))
            for field in fields
            if classified.get(field) is not None
        }
    else:
        lowered = str(failure_text or "").lower()
        if (
            "adapter_backend_analysis_failure" not in lowered
            and "adapter rejected backend analysis failure" not in lowered
        ):
            return None
        result = {
            "evidence_class": "instrument_defect",
            "category": "adapter_backend_analysis_failure",
            "action": "halt",
            "provider_code": "adapter_backend_analysis_failure",
            "retry_policy_kind": "terminal",
        }

    evidence_class = result.get("evidence_class", "unknown")
    category = result.get("category", "unclassified")
    class_label = evidence_class.replace("_", " ").capitalize()
    category_label = category.replace("_", " ")
    result["label"] = f"{class_label} / {category_label}"
    return result


def _classified_failure_action(classification: dict[str, str] | None) -> str | None:
    if not classification:
        return None
    evidence_class = classification.get("evidence_class")
    category = classification.get("category")
    action = classification.get("action")
    retry_policy = classification.get("retry_policy_kind")
    if evidence_class == "instrument_defect":
        return (
            "Stop paid calls. Fix the benchmark, adapter, or request configuration, "
            "run preflight again, then resume the same prepared contract."
        )
    if evidence_class == "model_signal":
        if action == "retry_bounded" or retry_policy in {"bounded_retry", "stochastic_retry"}:
            return (
                "Use only the recorded bounded retry policy. If it remains blocked, "
                "preserve it as model-signal evidence for review."
            )
        return (
            "Preserve this as model-signal evidence and route it through evidence "
            "review. Do not relabel it as a provider failure."
        )
    if evidence_class == "environment":
        if category == "billing":
            return (
                "Refill the provider account, clear any scheduler stop control, "
                "then resume the same prepared contract."
            )
        if action == "retry_bounded" or retry_policy in {"bounded_retry", "stochastic_retry"}:
            return (
                "Keep completed units, honor the shared cooldown and bounded retry "
                "policy, then resume the same prepared contract."
            )
        return (
            "Stop paid calls until provider health or credentials are restored, "
            "then resume the same prepared contract."
        )
    return (
        "Stop paid calls and review the classified event before retrying or "
        "dispositioning this unit."
    )


def _evidence_stage(
    status: dict[str, Any],
    event_name: Any | None = None,
    *,
    event_stage: Any | None = None,
    event_status: Any | None = None,
    event_validity: Any | None = None,
) -> str:
    status_text = str(event_status or status.get("status") or "")
    status_stage = str(status.get("stage") or "")
    stage = str(event_stage or "")
    validity = event_validity if event_validity is not None else status.get("validity")
    event_text = str(event_name or "")
    if "failed" in event_text or "incomplete" in event_text or status_text.startswith("failed"):
        return "attention"
    if event_text == "stage_completed" and status_text == "completed" and validity == "score_ready":
        return "score_ready"
    if event_text in {
        "score_saved",
        "score_reused",
        "result_saved",
        "score_skipped",
        "final_results_saved",
    }:
        return "score_ready"
    if "score" in event_text or event_text == "final_results_saved" or stage == "scoring":
        return "judging"
    if stage in {"generation", "run"} or "turn" in event_text or "conversation" in event_text:
        return "generating"
    if status_text == "completed" and validity == "score_ready":
        return "score_ready"
    if status_stage == "scoring":
        return "judging"
    if status_stage in {"generation", "run"}:
        return "generating"
    return stage or status_stage or "event"


def _score_state(
    status: dict[str, Any],
    classification: dict[str, str] | None = None,
) -> dict[str, str]:
    status_text = str(status.get("status") or "")
    validity_text = str(status.get("validity") or "")
    stage = str(status.get("stage") or "")
    module = str(status.get("module") or "")
    incomplete = status.get("incomplete_conversations")
    if not isinstance(incomplete, list):
        incomplete = []
    failure_text = " ".join(
        [
            str(status.get("failure_reason") or ""),
            *[str(item) for item in incomplete[:4]],
        ]
    )

    if status_text.startswith("failed"):
        classified_action = _classified_failure_action(classification)
        if classification and classification.get("evidence_class") == "instrument_defect":
            return {
                "kind": "blocked",
                "label": "Instrument defect",
                "detail": (
                    f"The failure was classified as {classification.get('category', 'instrument defect')}. "
                    "It is not a benchmark model outcome."
                ),
                "action": classified_action or "Stop paid calls and fix the recorded defect.",
            }
        if classification and classification.get("evidence_class") == "model_signal":
            return {
                "kind": "blocked",
                "label": "Model signal",
                "detail": (
                    f"The provider/model response was classified as "
                    f"{classification.get('category', 'model signal')}."
                ),
                "action": classified_action or "Preserve the evidence for review.",
            }
        if classification and classification.get("evidence_class") == "unknown":
            return {
                "kind": "blocked",
                "label": "Classification required",
                "detail": "The failure taxonomy could not determine a publishable disposition.",
                "action": classified_action or "Stop paid calls and inspect the classified event.",
            }
        if status_text == "failed_billing" or _looks_credit_exhausted(failure_text):
            return {
                "kind": "blocked",
                "label": "Credits exhausted",
                "detail": (
                    "The provider stopped paid calls because the account balance "
                    "or billing quota is exhausted."
                ),
                "action": (
                    "Refill the provider account, clear any scheduler stop control, "
                    "then resume the same prepared contract. Completed units will be reused."
                ),
            }
        if status_text == "failed_incomplete":
            if _looks_rate_limited(failure_text):
                return {
                    "kind": "blocked",
                    "label": "Rate-limited",
                    "detail": (
                        "Generation is incomplete because the provider returned a rate-limit response. "
                        "Scoring must wait until the exact missing item/side/turns are resumed cleanly."
                    ),
                    "action": "Wait for the shared rate-limit cooldown, then resume generation for the incomplete conversations. Do not score these partial transcripts.",
                }
            return {
                "kind": "blocked",
                "label": "Score blocked",
                "detail": (
                    "Generation is incomplete. Scoring commands should refuse this "
                    "directory until the provider, adapter, or model failure is fixed "
                    "and the same run is completed."
                ),
                "action": "Fix the model/provider path, then rerun generation. Do not score these partial transcripts.",
            }
        return {
            "kind": "blocked",
            "label": "Score blocked",
            "detail": "This terminal failure is diagnostic only and should not be promoted as benchmark evidence.",
            "action": "Fix the recorded failure, then rerun or resume from a clean completed ledger.",
        }

    if status_text == "completed" and validity_text == "score_ready":
        if stage == "generation":
            return {
                "kind": "ready",
                "label": "Ready for scoring",
                "detail": "Generation completed and the saved transcripts are valid inputs for the module score command.",
                "action": "Run the module scoring command when you are ready to spend judge calls.",
            }
        if stage == "scoring":
            return {
                "kind": "scored",
                "label": "Scored",
                "detail": "The scoring stage completed and score artifacts were written.",
                "action": "Review score files and reports before promoting the run.",
            }
        if module == "sus" and stage == "run":
            return {
                "kind": "scored",
                "label": "Scored inline",
                "detail": "SUS run stages include analyzer/judge scoring and wrote a summary artifact.",
                "action": "Review the SUS summary and conversation artifacts.",
            }
        return {
            "kind": "ready",
            "label": "Scored",
            "detail": "The ledger is complete and marked valid for downstream evidence review.",
            "action": "Inspect artifacts and reports before promoting the run.",
        }

    if status_text == "completed" and validity_text == "not_score_ready" and stage == "generation":
        return {
            "kind": "needs_scoring",
            "label": "Needs scoring",
            "detail": "Generation completed and the saved transcripts are ready for judge scoring.",
            "action": "Run the module scoring command when you are ready to spend judge calls.",
        }

    if validity_text == "not_score_ready":
        return {
            "kind": "blocked",
            "label": "Not scoreable",
            "detail": "The ledger is still running or incomplete, so scoring/promotion should wait.",
            "action": "Wait for completion or inspect the latest failure event.",
        }

    return {
        "kind": "unknown",
        "label": "Unknown",
        "detail": "The ledger did not expose enough status metadata to classify score readiness.",
        "action": "Inspect RUN_STATUS.json and RUN_EVENTS.jsonl directly.",
    }


def _event_digest(event: dict[str, Any], *, group: str, module_path: str) -> dict[str, Any]:
    return {
        "group": group,
        "module_path": module_path,
        "timestamp": event.get("timestamp"),
        "sequence": event.get("sequence"),
        "event": event.get("event"),
        "module": event.get("module"),
        "stage": event.get("stage"),
        "status": event.get("status"),
        "model": event.get("model"),
        "model_id": event.get("model_id"),
        "role": event.get("role"),
        "phase": event.get("phase"),
        "scenario": event.get("scenario"),
        "run_number": event.get("run_number"),
        "test_type": event.get("test_type"),
        "item_idx": event.get("item_idx"),
        "side": event.get("side"),
        "turn": event.get("turn"),
        "turns": event.get("turns"),
        "planned_turns": event.get("planned_turns"),
        "failure_stage": event.get("failure_stage"),
        "failure_reason": _shorten(event.get("failure_reason"), limit=180),
        "evidence_class": event.get("evidence_class"),
        "category": event.get("category"),
        "action": event.get("action"),
        "provider": event.get("provider"),
        "provider_code": event.get("provider_code"),
        "retry_policy_kind": event.get("retry_policy_kind"),
        "judge_model": event.get("judge_model"),
        "dimension": event.get("dimension"),
        "judge_result": event.get("judge_result"),
        "max_score": event.get("max_score"),
        "transcript_path": _display_path(event.get("transcript_path")),
        "score_path": _display_path(event.get("score_path")),
    }


def _event_sort_key(event: dict[str, Any]) -> tuple[str, int]:
    sequence = event.get("sequence")
    if not isinstance(sequence, int):
        sequence = -1
    timestamp = event.get("timestamp")
    return (timestamp if isinstance(timestamp, str) else "", sequence)


def _resolve_artifact_path(value: Any, output_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, output_dir / path]
    parts = path.parts
    if "results" in parts:
        candidates.insert(0, REPO_ROOT.joinpath(*parts[parts.index("results"):]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _preview_from_transcript_file(path: Path) -> dict[str, Any] | None:
    transcript = _load_json(path)
    turns = transcript.get("turns")
    if not isinstance(turns, list) or not turns:
        return None
    latest_turn = turns[-1]
    if not isinstance(latest_turn, dict):
        return None
    return {
        "path": _relative(path, REPO_ROOT),
        "turn": latest_turn.get("turn"),
        "user_message": _shorten(latest_turn.get("user_message"), limit=160),
        "model_response": _shorten(latest_turn.get("model_response"), limit=260),
        "model": transcript.get("model"),
        "item_idx": transcript.get("item_idx") or transcript.get("run_number"),
        "test_type": transcript.get("test_type") or transcript.get("scenario"),
        "side": transcript.get("side"),
        "completed": transcript.get("completed"),
    }


def _preview_from_sus_conversations_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    for result in reversed(data):
        if not isinstance(result, dict):
            continue
        conversation = result.get("conversation")
        if not isinstance(conversation, list) or not conversation:
            continue
        latest_user = None
        latest_assistant = None
        assistant_turn = 0
        for message in conversation:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if role == "user":
                latest_user = content
            elif role == "assistant":
                assistant_turn += 1
                latest_assistant = content
        if latest_user is None and latest_assistant is None:
            continue
        return {
            "path": _relative(path, REPO_ROOT),
            "turn": assistant_turn or None,
            "user_message": _shorten(latest_user, limit=160),
            "model_response": _shorten(latest_assistant, limit=260),
            "model": result.get("model"),
            "item_idx": result.get("run_number"),
            "test_type": result.get("scenario"),
            "side": None,
            "completed": True,
        }
    return None


def _conversation_path_candidates(status: dict[str, Any], output_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    metadata = status.get("metadata") if isinstance(status.get("metadata"), dict) else {}
    for value in metadata.get("source_files") or []:
        path = _resolve_artifact_path(value, output_dir)
        if path is not None:
            candidates.append(path)
    for value in (status.get("results_path"), status.get("summary_path")):
        path = _resolve_artifact_path(value, output_dir)
        if path is not None:
            candidates.append(path.with_name(f"{path.stem}-conversations.json"))
    candidates.extend(
        sorted(
            (path for path in output_dir.glob("*.json") if not path.name.startswith("RUN_")),
            key=lambda item: item.stat().st_mtime,
        )
    )
    candidates.extend(sorted(output_dir.glob("transcripts/*.json"), key=lambda item: item.stat().st_mtime))
    candidates.extend(sorted(output_dir.glob("*-conversations.json"), key=lambda item: item.stat().st_mtime))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _build_transcript_preview(
    events: list[dict[str, Any]],
    output_dir: Path,
    status: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    for event in reversed(events):
        path = _resolve_artifact_path(event.get("transcript_path"), output_dir)
        if path is None or not path.exists():
            continue
        preview = _preview_from_transcript_file(path)
        if preview is not None:
            return preview
    if status is not None:
        for path in reversed(_conversation_path_candidates(status, output_dir)):
            if not path.exists():
                continue
            preview = _preview_from_transcript_file(path)
            if preview is not None:
                return preview
            preview = _preview_from_sus_conversations_file(path)
            if preview is not None:
                return preview
    return None


def _transcript_preview(
    events: list[dict[str, Any]],
    output_dir: Path,
    status: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        directory_mtime = output_dir.stat().st_mtime_ns
    except OSError:
        directory_mtime = 0
    status = status or {}
    latest_event = events[-1] if events else {}
    path_key = str(output_dir.resolve())
    key = (
        path_key,
        directory_mtime,
        status.get("module"),
        status.get("stage"),
        status.get("status"),
        status.get("validity"),
        status.get("updated_at"),
        latest_event.get("sequence"),
        latest_event.get("timestamp"),
        latest_event.get("event"),
        len(events),
    )
    with _EVIDENCE_CACHE_LOCK:
        if key in _TRANSCRIPT_PREVIEW_CACHE:
            return _TRANSCRIPT_PREVIEW_CACHE[key]
    preview = _build_transcript_preview(events, output_dir, status)
    with _EVIDENCE_CACHE_LOCK:
        for old_key in [item for item in _TRANSCRIPT_PREVIEW_CACHE if item[0] == path_key]:
            _TRANSCRIPT_PREVIEW_CACHE.pop(old_key, None)
        _TRANSCRIPT_PREVIEW_CACHE[key] = preview
    return preview


def _turn_checks(user_message: Any, model_response: Any, turn: Any) -> dict[str, bool]:
    return {
        "json_ok": True,
        "user_ok": isinstance(user_message, str) and bool(user_message.strip()),
        "assistant_ok": isinstance(model_response, str) and bool(model_response.strip()),
        "turn_ok": isinstance(turn, int) or (isinstance(turn, str) and bool(turn.strip())),
        "provider_error": _looks_generic_model_error(model_response),
    }


def _evidence_item_problem(checks: dict[str, bool]) -> str | None:
    if not checks.get("json_ok"):
        return "Transcript JSON is malformed or unreadable."
    if not checks.get("user_ok") and not checks.get("assistant_ok"):
        return "No user or assistant text was saved for this turn."
    if not checks.get("user_ok"):
        return "User text is missing from the saved turn."
    if not checks.get("assistant_ok"):
        return "Assistant text is missing from the saved turn."
    if checks.get("provider_error"):
        return "Assistant text looks like a provider or adapter error."
    return None


def _evidence_item_from_turn(
    *,
    group: str,
    module_path: str,
    status: dict[str, Any],
    transcript: dict[str, Any],
    transcript_path: Path,
    turn: dict[str, Any],
    fallback_timestamp: Any,
) -> dict[str, Any]:
    checks = _turn_checks(turn.get("user_message"), turn.get("model_response"), turn.get("turn"))
    problem = _evidence_item_problem(checks)
    return {
        "kind": "turn_pair",
        "stage": "attention" if problem else _evidence_stage(status, "turn_saved", event_stage="generation"),
        "severity": "attention" if problem else _severity(status.get("status"), status.get("validity")),
        "timestamp": (
            turn.get("timestamp")
            or transcript.get("completed_at")
            or transcript.get("timestamp")
            or fallback_timestamp
        ),
        "group": group,
        "module": status.get("module") or module_path,
        "module_path": module_path,
        "status": status.get("status"),
        "validity": status.get("validity"),
        "model": transcript.get("model") or transcript.get("model_id"),
        "item_idx": transcript.get("item_idx") or transcript.get("run_number"),
        "test_type": transcript.get("test_type") or transcript.get("scenario"),
        "side": transcript.get("side"),
        "turn": turn.get("turn"),
        "planned_turns": (
            transcript.get("planned_num_turns")
            or transcript.get("num_turns")
            or transcript.get("actual_num_turns")
        ),
        "user_message": _shorten(turn.get("user_message"), limit=520),
        "model_response": _shorten(turn.get("model_response"), limit=700),
        "transcript_path": _relative(transcript_path, REPO_ROOT),
        "checks": checks,
        "problem": problem,
    }


def _evidence_item_from_turn_outcome(
    outcome: dict[str, Any],
    *,
    group: str,
    module_path: str,
    status: dict[str, Any],
    transcript: dict[str, Any],
    transcript_path: Path,
    fallback_timestamp: Any,
) -> dict[str, Any]:
    outcome_type = str(outcome.get("type") or "turn_outcome")
    stop_reason = str(outcome.get("stop_reason") or "unknown")
    problem = outcome_type.replace("_", " ").title()
    if outcome_type == "provider_refusal":
        problem = f"Provider refusal (stop reason: {stop_reason})"
    return {
        "kind": "turn_outcome",
        "event": outcome_type,
        "stage": "attention",
        "severity": "attention",
        "timestamp": outcome.get("timestamp") or fallback_timestamp,
        "group": group,
        "module": status.get("module") or module_path,
        "module_path": module_path,
        "status": status.get("status"),
        "validity": status.get("validity"),
        "model": transcript.get("model") or transcript.get("model_id"),
        "turn": outcome.get("turn"),
        "stop_reason": stop_reason,
        "problem": problem,
        "transcript_path": _relative(transcript_path, REPO_ROOT),
    }


def _build_evidence_items_from_transcript_file(
    path: Path,
    *,
    group: str,
    module_path: str,
    status: dict[str, Any],
    fallback_timestamp: Any,
    limit: int = 4,
) -> list[dict[str, Any]]:
    try:
        transcript = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return [
            {
                "kind": "turn_pair",
                "stage": "attention",
                "severity": "attention",
                "timestamp": fallback_timestamp,
                "group": group,
                "module": status.get("module") or module_path,
                "module_path": module_path,
                "status": status.get("status"),
                "validity": status.get("validity"),
                "transcript_path": _relative(path, REPO_ROOT),
                "checks": {"json_ok": False, "user_ok": False, "assistant_ok": False, "turn_ok": False, "provider_error": False},
                "problem": "Transcript JSON is malformed or unreadable.",
            }
        ]
    if not isinstance(transcript, dict):
        return []
    turns = transcript.get("turns")
    if not isinstance(turns, list):
        return []
    items: list[dict[str, Any]] = []
    for turn in turns[-limit:]:
        if not isinstance(turn, dict):
            continue
        items.append(
            _evidence_item_from_turn(
                group=group,
                module_path=module_path,
                status=status,
                transcript=transcript,
                transcript_path=path,
                turn=turn,
                fallback_timestamp=fallback_timestamp,
            )
        )
    outcomes = transcript.get("turn_outcomes")
    if isinstance(outcomes, list):
        for outcome in outcomes:
            if isinstance(outcome, dict):
                items.append(
                    _evidence_item_from_turn_outcome(
                        outcome,
                        group=group,
                        module_path=module_path,
                        status=status,
                        transcript=transcript,
                        transcript_path=path,
                        fallback_timestamp=fallback_timestamp,
                    )
                )
    return items[-limit:]


def _evidence_artifact_cache_key(
    kind: str,
    path: Path,
    *,
    group: str,
    module_path: str,
    status: dict[str, Any],
    fallback_timestamp: Any,
    limit: int,
) -> tuple[Any, ...] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (
        kind,
        str(path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
        group,
        module_path,
        status.get("module"),
        status.get("stage"),
        status.get("status"),
        status.get("validity"),
        str(fallback_timestamp or ""),
        limit,
    )


def _store_evidence_artifact_cache(
    key: tuple[Any, ...],
    items: list[dict[str, Any]],
) -> None:
    artifact_identity = key[:2]
    with _EVIDENCE_CACHE_LOCK:
        stale_keys = [
            cached_key
            for cached_key in _EVIDENCE_ARTIFACT_CACHE
            if cached_key[:2] == artifact_identity and cached_key != key
        ]
        for stale_key in stale_keys:
            _EVIDENCE_ARTIFACT_CACHE.pop(stale_key, None)
        _EVIDENCE_ARTIFACT_CACHE[key] = items


def _evidence_items_from_transcript_file(
    path: Path,
    *,
    group: str,
    module_path: str,
    status: dict[str, Any],
    fallback_timestamp: Any,
    limit: int = 4,
) -> list[dict[str, Any]]:
    key = _evidence_artifact_cache_key(
        "transcript", path, group=group, module_path=module_path, status=status,
        fallback_timestamp=fallback_timestamp, limit=limit,
    )
    if key is not None:
        with _EVIDENCE_CACHE_LOCK:
            cached = _EVIDENCE_ARTIFACT_CACHE.get(key)
        if cached is not None:
            return cached
    items = _build_evidence_items_from_transcript_file(
        path, group=group, module_path=module_path, status=status,
        fallback_timestamp=fallback_timestamp, limit=limit,
    )
    if key is not None:
        _store_evidence_artifact_cache(key, items)
    return items


def _build_evidence_items_from_sus_conversations_file(
    path: Path,
    *,
    group: str,
    module_path: str,
    status: dict[str, Any],
    fallback_timestamp: Any,
    limit: int = 4,
) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return [
            {
                "kind": "turn_pair",
                "stage": "attention",
                "severity": "attention",
                "timestamp": fallback_timestamp,
                "group": group,
                "module": status.get("module") or module_path,
                "module_path": module_path,
                "status": status.get("status"),
                "validity": status.get("validity"),
                "transcript_path": _relative(path, REPO_ROOT),
                "checks": {"json_ok": False, "user_ok": False, "assistant_ok": False, "turn_ok": False, "provider_error": False},
                "problem": "Conversation JSON is malformed or unreadable.",
            }
        ]
    if not isinstance(data, list):
        return []
    items: list[dict[str, Any]] = []
    for result in data[-limit:]:
        if not isinstance(result, dict):
            continue
        conversation = result.get("conversation")
        if not isinstance(conversation, list):
            continue
        latest_user = None
        assistant_turn = 0
        for message in conversation:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                latest_user = content
            elif role == "assistant":
                assistant_turn += 1
                turn = {
                    "turn": assistant_turn,
                    "user_message": latest_user,
                    "model_response": content,
                }
                items.append(
                    _evidence_item_from_turn(
                        group=group,
                        module_path=module_path,
                        status=status,
                        transcript={
                            "model": result.get("model"),
                            "run_number": result.get("run_number"),
                            "scenario": result.get("scenario"),
                            "completed": True,
                            "num_turns": assistant_turn,
                        },
                        transcript_path=path,
                        turn=turn,
                        fallback_timestamp=fallback_timestamp,
                    )
                )
        outcomes = result.get("turn_outcomes")
        if isinstance(outcomes, list):
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue
                items.append(
                    _evidence_item_from_turn_outcome(
                        outcome,
                        group=group,
                        module_path=module_path,
                        status=status,
                        transcript=result,
                        transcript_path=path,
                        fallback_timestamp=fallback_timestamp,
                    )
                )
    return items[-limit:]


def _evidence_items_from_sus_conversations_file(
    path: Path,
    *,
    group: str,
    module_path: str,
    status: dict[str, Any],
    fallback_timestamp: Any,
    limit: int = 4,
) -> list[dict[str, Any]]:
    key = _evidence_artifact_cache_key(
        "sus", path, group=group, module_path=module_path, status=status,
        fallback_timestamp=fallback_timestamp, limit=limit,
    )
    if key is not None:
        with _EVIDENCE_CACHE_LOCK:
            cached = _EVIDENCE_ARTIFACT_CACHE.get(key)
        if cached is not None:
            return cached
    items = _build_evidence_items_from_sus_conversations_file(
        path, group=group, module_path=module_path, status=status,
        fallback_timestamp=fallback_timestamp, limit=limit,
    )
    if key is not None:
        _store_evidence_artifact_cache(key, items)
    return items


def _build_evidence_items_from_module(
    *,
    group: str,
    module_path: str,
    status: dict[str, Any],
    events: list[dict[str, Any]],
    output_dir: Path,
    limit: int = MAX_MODULE_EVIDENCE_ITEMS,
) -> list[dict[str, Any]]:
    latest_event = events[-1] if events else {}
    fallback_timestamp = status.get("updated_at") or latest_event.get("timestamp")
    items: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for event in reversed(events):
        path = _resolve_artifact_path(event.get("transcript_path"), output_dir)
        if path is None or not path.exists() or path.resolve() in seen_paths:
            continue
        seen_paths.add(path.resolve())
        items.extend(
            _evidence_items_from_transcript_file(
                path,
                group=group,
                module_path=module_path,
                status=status,
                fallback_timestamp=event.get("timestamp") or fallback_timestamp,
                limit=MAX_TRANSCRIPT_EVIDENCE_TURNS,
            )
        )
        if len(items) >= limit:
            break
    if len(items) < limit:
        for path in reversed(_conversation_path_candidates(status, output_dir)):
            if not path.exists() or path.resolve() in seen_paths:
                continue
            seen_paths.add(path.resolve())
            transcript_items = _evidence_items_from_transcript_file(
                path,
                group=group,
                module_path=module_path,
                status=status,
                fallback_timestamp=fallback_timestamp,
                limit=MAX_TRANSCRIPT_EVIDENCE_TURNS,
            )
            conversation_items = _evidence_items_from_sus_conversations_file(
                path,
                group=group,
                module_path=module_path,
                status=status,
                fallback_timestamp=fallback_timestamp,
                limit=MAX_TRANSCRIPT_EVIDENCE_TURNS,
            )
            items.extend(transcript_items or conversation_items)
            if len(items) >= limit:
                break
    evidence_events = [
        event for event in events if event.get("event") in EVIDENCE_LEDGER_EVENTS
    ]
    for event in evidence_events[-limit:]:
        name = event.get("event")
        event_stage = _evidence_stage(
            status,
            name,
            event_stage=event.get("stage"),
            event_status=event.get("status"),
            event_validity=event.get("validity"),
        )
        items.append(
            {
                "kind": "event",
                "stage": event_stage,
                "severity": "attention" if event_stage == "attention" else _severity(status.get("status"), status.get("validity")),
                "timestamp": event.get("timestamp") or fallback_timestamp,
                "group": group,
                "module": event.get("module") or status.get("module") or module_path,
                "module_path": module_path,
                "status": event.get("status") or status.get("status"),
                "validity": status.get("validity"),
                "event": name,
                "model": event.get("model"),
                "item_idx": event.get("item_idx"),
                "test_type": event.get("test_type") or event.get("scenario"),
                "side": event.get("side"),
                "turn": event.get("turn"),
                "planned_turns": event.get("planned_turns") or event.get("turns"),
                "judge_model": event.get("judge_model"),
                "dimension": event.get("dimension"),
                "judge_result": event.get("judge_result"),
                "max_score": event.get("max_score"),
                "score_path": _display_path(event.get("score_path")),
                "transcript_path": _display_path(event.get("transcript_path")),
                "problem": _shorten(event.get("failure_reason"), limit=220),
                "evidence_class": event.get("evidence_class"),
                "category": event.get("category"),
                "action": event.get("action"),
                "provider": event.get("provider"),
                "provider_code": event.get("provider_code"),
                "retry_policy_kind": event.get("retry_policy_kind"),
            }
        )
    return sorted(items, key=_event_sort_key)[-limit:]


def _evidence_items_from_module(
    *,
    group: str,
    module_path: str,
    status: dict[str, Any],
    events: list[dict[str, Any]],
    output_dir: Path,
    limit: int = MAX_MODULE_EVIDENCE_ITEMS,
) -> list[dict[str, Any]]:
    try:
        directory_mtime = output_dir.stat().st_mtime_ns
    except OSError:
        directory_mtime = 0
    latest_event = events[-1] if events else {}
    path_key = str(output_dir.resolve())
    key = (
        path_key,
        directory_mtime,
        group,
        module_path,
        status.get("module"),
        status.get("stage"),
        status.get("status"),
        status.get("validity"),
        status.get("updated_at"),
        latest_event.get("sequence"),
        latest_event.get("timestamp"),
        latest_event.get("event"),
        len(events),
        limit,
    )
    with _EVIDENCE_CACHE_LOCK:
        cached = _MODULE_EVIDENCE_CACHE.get(key)
    if cached is not None:
        return cached
    items = _build_evidence_items_from_module(
        group=group,
        module_path=module_path,
        status=status,
        events=events,
        output_dir=output_dir,
        limit=limit,
    )
    with _EVIDENCE_CACHE_LOCK:
        for old_key in [item for item in _MODULE_EVIDENCE_CACHE if item[0] == path_key]:
            _MODULE_EVIDENCE_CACHE.pop(old_key, None)
        _MODULE_EVIDENCE_CACHE[key] = items
    return items


def _attention_summary(status: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    severity = _severity(status.get("status"), status.get("validity"))
    if severity != "attention":
        return None

    incomplete = status.get("incomplete_conversations")
    if not isinstance(incomplete, list):
        incomplete = []
    incomplete_examples = [
        item for item in (_shorten(value, limit=110) for value in incomplete[:4]) if item
    ]

    failed_events = [
        event
        for event in events
        if str(event.get("event") or "").endswith("_failed")
        or event.get("event") == "conversation_incomplete"
    ]
    latest_failed = failed_events[-1] if failed_events else {}
    reason = (
        _shorten(status.get("failure_reason"), limit=240)
        or _shorten(latest_failed.get("failure_reason"), limit=240)
        or "No failure reason was recorded."
    )

    failure_text = " ".join([str(reason), *incomplete_examples])
    classification = _failure_classification(events, failure_text)
    title = "Needs review"
    if status.get("status") == "failed_billing" or _looks_credit_exhausted(failure_text):
        title = "Credits exhausted - refill required"
    elif _looks_rate_limited(reason) or any(_looks_rate_limited(value) for value in incomplete_examples):
        title = "Rate-limited incomplete generation"
    elif status.get("status") == "failed_incomplete":
        title = "Incomplete generation"
    elif str(status.get("status") or "").startswith("failed"):
        title = str(status.get("status")).replace("_", " ")
    elif status.get("validity") == "not_score_ready":
        title = "Not scoreable"

    summary = {
        "title": title,
        "reason": reason,
        "incomplete_count": len(incomplete),
        "incomplete_examples": incomplete_examples,
        "failure_stage": latest_failed.get("failure_stage"),
        "latest_failure_event": latest_failed.get("event"),
    }
    if classification:
        summary["classification"] = classification
        summary["action"] = _classified_failure_action(classification)
    return summary


def _event_progress(events: list[dict[str, Any]], status: dict[str, Any]) -> dict[str, Any]:
    turn_saved = 0
    planned_turns = 0
    conversations_started = 0
    conversations_incomplete = 0
    scores_saved = 0
    scores_skipped = 0
    final_results_saved = 0
    judge_calls_completed = 0
    failures = 0

    for event in events:
        name = event.get("event")
        if name == "conversation_started":
            conversations_started += 1
            planned = event.get("planned_turns")
            if isinstance(planned, int) and planned > 0:
                planned_turns += planned
        elif name == "conversation_incomplete":
            conversations_incomplete += 1
        elif name == "turn_saved":
            turn_saved += 1
        elif name == "score_saved":
            scores_saved += 1
        elif name == "score_skipped":
            scores_skipped += 1
        elif name == "final_results_saved":
            final_results_saved += 1
        elif (
            name == "paid_call_completed"
            and event.get("stage") == "scoring"
            and event.get("role") == "judge"
        ):
            judge_calls_completed += 1
        elif name in {"stage_failed", "conversation_failed", "score_failed"}:
            failures += 1

    # Deduplicate by canonical unit_id across attempts.
    conversations_completed = len(completed_unit_keys(events))

    terminal = status.get("status") in {
        "completed",
        "failed_auth",
        "failed_billing",
        "failed_incomplete",
        "failed_invalid",
        "failed_rate_limited",
        "failed_provider",
        "failed_scoring",
        "failed_timeout",
    }
    if terminal:
        percent = 100
    elif planned_turns:
        generation = min(turn_saved / planned_turns, 1.0) * 80
        scoring = min(scores_saved, max(conversations_started, 1)) / max(conversations_started, 1) * 20
        percent = round(min(generation + scoring, 99))
    else:
        percent = None

    return {
        "percent": percent,
        "turn_saved": turn_saved,
        "planned_turns": planned_turns,
        "conversations_started": conversations_started,
        "conversations_completed": conversations_completed,
        "conversations_incomplete": conversations_incomplete,
        "scores_saved": scores_saved,
        "scores_skipped": scores_skipped,
        "final_results_saved": final_results_saved,
        "judge_calls_completed": judge_calls_completed,
        "failures": failures,
    }


def _contract_modules_error(contract: dict[str, Any]) -> str | None:
    modules = contract.get("modules")
    if not isinstance(modules, list):
        return "modules must be a list"
    for module in modules:
        if not isinstance(module, dict):
            return "modules entries must be objects"
        if not isinstance(module.get("expected_units"), list):
            return "expected_units must be a list"
    return None


def _contract_artifact_directory_signature(
    contract: dict[str, Any],
    *,
    contract_path: Path,
    results_root: Path,
) -> tuple[Any, ...]:
    """Track directories whose child presence affects a contract summary."""
    contract_dir = contract_path.parent
    directories: set[Path] = {contract_dir}

    def resolve(value: Any, *, output_dir: Path | None = None) -> Path | None:
        if not isinstance(value, str) or not value.strip():
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        candidates = []
        if output_dir is not None:
            candidates.append(output_dir / path)
        candidates.extend((results_root / path, contract_dir / path, REPO_ROOT / path))
        return next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    def track_artifact(value: Any, *, output_dir: Path) -> None:
        artifact = resolve(value, output_dir=output_dir)
        if artifact is None:
            return
        parent = artifact.parent
        directories.add(parent)
        # Missing nested directories become observable when their closest
        # existing ancestor receives the new child directory.
        ancestor = parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
            directories.add(ancestor)

    for module in contract.get("modules") or []:
        if not isinstance(module, dict):
            continue
        output_dir = resolve(module.get("output_dir")) or contract_dir
        directories.add(output_dir)
        for unit in module.get("expected_units") or []:
            if not isinstance(unit, dict):
                continue
            for field in (
                "expected_transcript_path",
                "expected_score_path",
                "expected_summary_path",
                "expected_trace_path",
            ):
                track_artifact(unit.get(field), output_dir=output_dir)
        for artifact in module.get("expected_artifacts") or []:
            if isinstance(artifact, dict):
                track_artifact(artifact.get("path"), output_dir=output_dir)

    signature: list[Any] = []
    for directory in sorted(directories, key=lambda item: item.as_posix()):
        try:
            stat = directory.stat()
            signature.append((directory.as_posix(), stat.st_mtime_ns))
        except OSError:
            signature.append((directory.as_posix(), None))
    return tuple(signature)


def _load_contract_summaries(
    root: Path,
    *,
    warnings: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for contract_path in _iter_ledger_paths(root, CONTRACT_FILENAME):
        try:
            contract_stat = contract_path.stat()
            directory_stat = contract_path.parent.stat()
            signature = (
                contract_stat.st_mtime_ns,
                contract_stat.st_size,
                directory_stat.st_mtime_ns,
            )
        except OSError:
            continue
        contract = load_run_contract(contract_path)
        if not contract:
            continue
        contract_error = _contract_modules_error(contract)
        if contract_error:
            if warnings is not None:
                warnings.append({
                    "kind": "run_contract",
                    "path": _relative(contract_path, REPO_ROOT),
                    "error": contract_error,
                })
            continue
        signature += _contract_artifact_directory_signature(
            contract,
            contract_path=contract_path,
            results_root=root,
        )
        cache_key = str(contract_path.resolve())
        with _CONTRACT_CACHE_LOCK:
            cached = _CONTRACT_SUMMARY_CACHE.get(cache_key)
        if cached and cached[0] == signature:
            summary = copy.deepcopy(cached[1])
        else:
            raw_modules = [
                module
                for module in contract.get("modules") or []
                if isinstance(module, dict)
            ]
            raw_module_models = [
                _expected_model_unit_breakdown(module)
                for module in raw_modules
            ]
            summary = summarize_contract(contract, contract_path=contract_path, results_root=root)
            call_plan = contract.get("call_plan")
            if isinstance(call_plan, dict):
                lines = call_plan.get("lines") if isinstance(call_plan.get("lines"), list) else []
                summary["call_plan"] = sanitize_ledger_value({
                    "schema_version": call_plan.get("schema_version"),
                    "total_calls": call_plan.get("total_calls"),
                    "line_count": len(lines),
                    "scoring_judge_calls_expected": _scoring_judge_call_count(call_plan),
                })
            estimate = contract.get("cost_estimate")
            if isinstance(estimate, dict):
                summary["cost_estimate"] = sanitize_ledger_value({
                    key: estimate.get(key)
                    for key in (
                        "schema_version", "state", "total_cost_usd", "cost_by_stage",
                        "cost_by_role", "cost_by_provider", "unknown_pricing", "notice",
                    )
                    if estimate.get(key) is not None
                })
            module_count = len(raw_modules)
            for module_summary, raw_module, model_breakdown in zip(
                summary.get("modules") or [], raw_modules, raw_module_models
            ):
                if isinstance(module_summary, dict):
                    score_projection = _score_unit_projection(raw_module)
                    module_summary["models"] = model_breakdown
                    module_summary.update(score_projection)
                    module_summary["judge_calls_expected"] = _scoring_judge_call_count(
                        call_plan,
                        module_name=raw_module.get("module"),
                        module_count=module_count,
                    )
            for key in ("path", "results_root", "output_dir", "run_dir", "contract_path"):
                if key in summary:
                    summary[key] = _display_path(summary.get(key))
            if isinstance(summary.get("expected_artifacts"), list):
                summary["expected_artifacts"] = _display_paths_in_value(summary["expected_artifacts"])
            if isinstance(summary.get("present_artifacts"), list):
                summary["present_artifacts"] = _display_paths_in_value(summary["present_artifacts"])
            if isinstance(summary.get("missing_required_artifacts"), list):
                summary["missing_required_artifacts"] = _display_paths_in_value(summary["missing_required_artifacts"])
            try:
                rel_parts = contract_path.parent.resolve().relative_to(root.resolve()).parts
                if rel_parts:
                    summary["path_group_id"] = rel_parts[0]
            except (OSError, ValueError):
                pass
            with _CONTRACT_CACHE_LOCK:
                _CONTRACT_SUMMARY_CACHE[cache_key] = (signature, copy.deepcopy(summary))
        control_path = contract_path.parent / CONTROL_FILENAME
        summary["control"] = summarize_control(
            load_run_control(control_path),
            control_path=control_path,
        )
        summaries.append(summary)
    return summaries


def _load_contract_headers(
    root: Path,
    *,
    warnings: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Load contract metadata without resolving every expected artifact."""
    headers: list[dict[str, Any]] = []
    for contract_path in _iter_ledger_paths(root, CONTRACT_FILENAME):
        try:
            contract_stat = contract_path.stat()
        except OSError:
            continue
        signature = (contract_stat.st_mtime_ns, contract_stat.st_size)
        cache_key = str(contract_path.resolve())
        with _CONTRACT_CACHE_LOCK:
            cached = _CONTRACT_HEADER_CACHE.get(cache_key)
        if cached and cached[0] == signature:
            header = copy.deepcopy(cached[1])
            control_path = contract_path.parent / CONTROL_FILENAME
            header["control"] = summarize_control(
                load_run_control(control_path),
                control_path=control_path,
            )
            headers.append(header)
            continue

        contract = load_run_contract(contract_path)
        if not contract:
            continue
        contract_error = _contract_modules_error(contract)
        if contract_error:
            if warnings is not None:
                warnings.append({
                    "kind": "run_contract",
                    "path": _relative(contract_path, REPO_ROOT),
                    "error": contract_error,
                })
            continue
        modules = []
        for raw_module in contract.get("modules") or []:
            if not isinstance(raw_module, dict):
                continue
            units = [item for item in raw_module.get("expected_units") or [] if isinstance(item, dict)]
            score_projection = _score_unit_projection(raw_module)
            modules.append({
                "module": raw_module.get("module"),
                "stage": raw_module.get("stage"),
                "output_dir": _display_path(raw_module.get("output_dir")),
                "expected_units": len(units),
                **score_projection,
                "complete_units": 0,
                "missing_units": len(units),
                "models": _expected_model_unit_breakdown(raw_module),
            })
        expected_units = sum(item["expected_units"] for item in modules)
        lifecycle_state = contract.get("lifecycle_state") or contract.get("state")
        expected_models = [
            {
                **{
                    key: model.get(key)
                    for key in ("key", "label", "model_id", "endpoint")
                    if model.get(key) is not None
                },
                **(
                    {"adapter_profile": model["condition_metadata"]["adapter_profile"]}
                    if isinstance(model.get("condition_metadata"), dict)
                    and model["condition_metadata"].get("adapter_profile")
                    else {}
                ),
            }
            for model in contract.get("expected_models") or []
            if isinstance(model, dict)
        ]
        expected_judges = [
            {
                key: judge.get(key)
                for key in ("role", "label", "model_id", "endpoint")
                if judge.get(key) is not None
            }
            for judge in contract.get("expected_judges") or []
            if isinstance(judge, dict)
        ]
        raw_identity = (
            contract.get("identity")
            if isinstance(contract.get("identity"), dict)
            else {}
        )
        raw_sample_spec = (
            raw_identity.get("sample_spec")
            if isinstance(raw_identity.get("sample_spec"), dict)
            else {}
        )
        sample_items = (
            raw_sample_spec.get("item_indices")
            or raw_sample_spec.get("items")
            or []
        )
        sample_spec = {
            key: raw_sample_spec.get(key)
            for key in ("scenario_ids", "runs")
            if raw_sample_spec.get(key) is not None
        }
        module_names = {
            _normalized_benchmark_module(module.get("module"))
            for module in modules
        }
        if (
            isinstance(sample_items, dict)
            and "epistemic" in module_names
            and all(isinstance(items, list) for items in sample_items.values())
        ):
            sample_spec["case_count"] = sum(
                len(items)
                for items in sample_items.values()
                if isinstance(items, list)
            )
            sample_spec["test_type_count"] = len(sample_items)
            sample_spec["conversation_count"] = expected_units
        elif isinstance(sample_items, (list, tuple, dict)):
            sample_spec["item_count"] = len(sample_items)
        raw_provenance = (
            contract.get("provenance")
            if isinstance(contract.get("provenance"), dict)
            else {}
        )
        provenance = {
            key: raw_provenance.get(key)
            for key in (
                "benchmark_condition_hash",
                "sample_condition_hash",
                "comparison_spec_hash",
                "model_conditions_hash",
                "batch_condition_hash",
            )
            if raw_provenance.get(key) is not None
        }
        header = {
            "present": True,
            "schema_version": contract.get("schema_version"),
            "path": _display_path(contract_path),
            "run_id": contract.get("run_id") or contract_path.parent.name,
            "contract_scope": contract.get("contract_scope"),
            "lifecycle_state": lifecycle_state,
            "prepared": lifecycle_state == "prepared" or contract.get("prepared") is True,
            "created_at": contract.get("created_at"),
            "source_command": contract.get("source_command"),
            "execute_command": contract.get("execute_command"),
            "scheduler_command": _scheduler_run_command(contract_path),
            "results_root": _display_path(contract.get("results_root")),
            "model_selector": contract.get("model_selector"),
            "judge_set": contract.get("judge_set"),
            "expected_models": expected_models,
            "expected_judges": expected_judges,
            "modules": modules,
            "expected_units": expected_units,
            "complete_units": 0,
            "missing_units": expected_units,
            "missing_required_artifacts": [],
            "model_mismatches": [],
            "attention": False,
            "progress_percent": 0 if expected_units else 100,
            "identity": {"sample_spec": sample_spec} if sample_spec else {},
            "provenance": provenance,
            "contract_fingerprint": contract.get("fingerprint"),
            "fingerprint": contract.get("fingerprint"),
        }
        call_plan = contract.get("call_plan")
        if isinstance(call_plan, dict):
            lines = call_plan.get("lines") if isinstance(call_plan.get("lines"), list) else []
            header["call_plan"] = {
                "schema_version": call_plan.get("schema_version"),
                "total_calls": call_plan.get("total_calls"),
                "line_count": len(lines),
                "scoring_judge_calls_expected": _scoring_judge_call_count(call_plan),
            }
            for module in modules:
                module["judge_calls_expected"] = _scoring_judge_call_count(
                    call_plan,
                    module_name=module.get("module"),
                    module_count=len(modules),
                )
        estimate = contract.get("cost_estimate")
        if isinstance(estimate, dict):
            header["cost_estimate"] = {
                key: estimate.get(key)
                for key in (
                    "schema_version", "state", "total_cost_usd", "cost_by_stage",
                    "cost_by_role", "cost_by_provider", "unknown_pricing", "notice",
                )
                if estimate.get(key) is not None
            }
        try:
            rel_parts = contract_path.parent.resolve().relative_to(root.resolve()).parts
            if rel_parts:
                header["path_group_id"] = rel_parts[0]
        except (OSError, ValueError):
            pass
        header = sanitize_ledger_value(header)
        with _CONTRACT_CACHE_LOCK:
            _CONTRACT_HEADER_CACHE[cache_key] = (signature, copy.deepcopy(header))
        control_path = contract_path.parent / CONTROL_FILENAME
        header["control"] = summarize_control(
            load_run_control(control_path),
            control_path=control_path,
        )
        headers.append(header)
    return headers


def _load_run_plans(
    root: Path,
    *,
    warnings: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for plan_path in _iter_ledger_paths(root, PLAN_FILENAME):
        plan = load_run_plan(plan_path)
        if not plan:
            continue
        raw_modules = plan.get("modules")
        if not isinstance(raw_modules, list):
            if warnings is not None:
                warnings.append({
                    "kind": "run_plan",
                    "path": _relative(plan_path, REPO_ROOT),
                    "error": "modules must be a list",
                })
            continue
        invalid_units = any(
            isinstance(module, dict)
            and not (
                isinstance(module.get("expected_units"), (list, dict))
                or (
                    isinstance(module.get("expected_units"), int)
                    and not isinstance(module.get("expected_units"), bool)
                    and module.get("expected_units") >= 0
                )
                or module.get("expected_units") is None
            )
            for module in raw_modules
        )
        if invalid_units:
            if warnings is not None:
                warnings.append({
                    "kind": "run_plan",
                    "path": _relative(plan_path, REPO_ROOT),
                    "error": "expected_units must be a non-negative integer or collection",
                })
            continue
        summary = {
            "present": True,
            "schema_version": plan.get("schema_version"),
            "run_id": plan.get("run_id") or plan_path.parent.name,
            "path": _relative(plan_path, REPO_ROOT),
            "lifecycle_state": plan.get("lifecycle_state") or plan.get("state"),
            "created_at": plan.get("created_at"),
            "updated_at": plan.get("updated_at") or plan.get("created_at"),
            "results_root": _display_path(plan.get("results_root")),
            "model_selector": plan.get("model_selector"),
            "judge_set": plan.get("judge_set"),
            "modules": raw_modules,
            "completion_gates": list(plan.get("completion_gates") or [])
            if isinstance(plan.get("completion_gates"), (list, tuple)) else [],
            "source_command": plan.get("source_command"),
        }
        summary["expected_units"] = sum(
            _flow_unit_count(module.get("expected_units"))
            for module in summary["modules"]
            if isinstance(module, dict)
        )
        try:
            rel_parts = plan_path.parent.resolve().relative_to(root.resolve()).parts
            if rel_parts:
                summary["path_group_id"] = rel_parts[0]
        except (OSError, ValueError):
            pass
        plans.append(summary)
    return plans


def _load_scheduler_summaries(root: Path) -> list[dict[str, Any]]:
    """Load process-level scheduler state files under the dashboard root."""
    schedulers: list[dict[str, Any]] = []
    for status_path in _iter_ledger_paths(root, SCHEDULER_STATUS_FILENAME):
        status = sanitize_ledger_value(load_scheduler_status(status_path))
        if not status:
            continue
        try:
            rel_parts = status_path.parent.resolve().relative_to(root.resolve()).parts
        except (OSError, ValueError):
            rel_parts = ()
        path_group_id = rel_parts[0] if rel_parts else status_path.parent.name
        disposition = _load_disposition(status_path.parent)
        rejected_from_analysis = _is_rejected_from_analysis(disposition)
        progress = status.get("progress") if isinstance(status.get("progress"), dict) else {}
        settings = status.get("settings") if isinstance(status.get("settings"), dict) else {}
        runner = status.get("runner") if isinstance(status.get("runner"), dict) else {}
        contract = status.get("contract") if isinstance(status.get("contract"), dict) else {}
        control = status.get("control") if isinstance(status.get("control"), dict) else {}
        state = status.get("state")
        raw_state = state
        module_status = sanitize_ledger_value(_load_json(status_path.parent / "RUN_STATUS.json"))
        if (
            state in {"attention", "stopped"}
            and isinstance(module_status, dict)
            and module_status.get("status") == "completed"
            and module_status.get("validity") == "score_ready"
        ):
            state = "score_ready"
            runner = {
                **runner,
                "status": module_status.get("status"),
                "stage": module_status.get("stage"),
                "validity": module_status.get("validity"),
                "updated_at": module_status.get("updated_at"),
                "failure_reason": None,
            }
        if rejected_from_analysis:
            state = "rejected"
        schedulers.append(
            {
                "present": True,
                "schema_version": status.get("schema_version"),
                "scheduler_id": status.get("scheduler_id"),
                "state": state,
                "raw_state": raw_state,
                "analysis_state": disposition.get("disposition") or "candidate",
                "disposition": disposition,
                "run_id": status.get("run_id") or path_group_id,
                "path": _relative(status_path, REPO_ROOT),
                "path_group_id": path_group_id,
                "contract_path": _display_path(status.get("contract_path")),
                "state_dir": _display_path(status.get("state_dir")),
                "command": status.get("command"),
                "score_command": status.get("score_command"),
                "created_at": status.get("created_at"),
                "started_at": status.get("started_at"),
                "updated_at": status.get("updated_at"),
                "completed_at": status.get("completed_at"),
                "reason": status.get("reason"),
                "settings": settings,
                "runner": runner,
                "contract": contract,
                "control": control,
                "progress": progress,
                "expected_units": progress.get("expected_units") or contract.get("expected_units") or 0,
                "complete_units": progress.get("completed_units") or contract.get("complete_units") or 0,
                "active_units": progress.get("active_units") or 0,
                "eta_seconds": progress.get("eta_seconds"),
                "eta_basis": progress.get("eta_basis"),
                "average_completed_unit_seconds": progress.get("average_completed_unit_seconds"),
                "max_active_calls": settings.get("max_active_calls"),
            }
        )
    return schedulers


FLOW_LANES = [
    ("prepared", "Prepared", "Contracts exist; no paid artifacts yet."),
    ("queued", "Queued", "Scheduler has accepted work but no runner process is active."),
    ("generating", "Generating", "Model-under-test work is active or just starting."),
    ("needs_scoring", "Needs Scoring", "Generation finished; judge artifacts are still expected."),
    ("scoring", "Scoring", "Judge or score aggregation work is active."),
    ("score_ready", "Scored", "Score artifacts are ready for review."),
    ("attention", "Attention", "Blocked, stale, failed, or contract-mismatch work."),
    ("rejected", "Rejected", "Malformed diagnostic runs excluded from analysis."),
]


def _is_openrouter_endpoint(endpoint: str) -> bool:
    normalized = str(endpoint or "").strip().rstrip("/")
    return normalized in {"openrouter", "https://openrouter.ai/api/v1"}


def _flow_lane_for_module(module: dict[str, Any]) -> str:
    status = str(module.get("status") or "")
    stage = str(module.get("stage") or "")
    validity = str(module.get("validity") or "")
    if _is_rejected_from_analysis(module.get("disposition") or {}):
        return "rejected"
    if module.get("severity") == "attention" or status.startswith("failed") or module.get("attention"):
        return "attention"
    if module.get("severity") == "running" or status == "running":
        return "scoring" if stage == "scoring" else "generating"
    if validity == "score_ready":
        return "score_ready"
    if status == "completed":
        return "needs_scoring"
    return "generating"


def _flow_next_action(lane_id: str, module: dict[str, Any] | None = None) -> str:
    if lane_id == "prepared":
        return "Review the plan and copy the execute command when ready."
    if lane_id == "generating":
        return "Watch heartbeat, latest artifact writes, and spend before scoring."
    if lane_id == "needs_scoring":
        return "Run scoring only after generation is complete and not incomplete."
    if lane_id == "scoring":
        return "Watch judge writes and panel completion."
    if lane_id == "score_ready":
        return "Review scored artifacts and promotion checks."
    if lane_id == "rejected":
        return "Excluded from scored analysis; keep as audit evidence only."
    if module:
        attention = module.get("attention") or {}
        score_state = module.get("score_state") or {}
        return attention.get("action") or score_state.get("action") or "Inspect and resolve before spending more calls."
    return "Inspect and resolve before spending more calls."


DEFERRED_SCORE_ARTIFACT_KINDS = {"final_results", "report"}


def _contract_attention_for_dashboard(contract: dict[str, Any], modules: list[dict[str, Any]]) -> bool:
    """Return whether a contract should show as an operator attention item.

    Contract summaries correctly treat missing promotion artifacts as blocking
    final promotion. In the live dashboard, a clean generation that is
    intentionally waiting on scoring should remain in Needs Scoring instead of
    looking broken.
    """
    if not contract.get("attention"):
        return False
    if modules and all(_is_rejected_from_analysis(module.get("disposition") or {}) for module in modules):
        return False
    if contract.get("model_mismatches"):
        return True

    missing = [
        item for item in contract.get("missing_required_artifacts") or []
        if isinstance(item, dict)
    ]
    if not missing:
        return False

    waiting_for_scoring = any(
        ((module.get("score_state") or {}).get("kind") == "needs_scoring")
        for module in modules
    )
    if not waiting_for_scoring:
        return True

    return not all(
        str(item.get("kind") or "").lower() in DEFERRED_SCORE_ARTIFACT_KINDS
        and str(item.get("required_for") or "").lower() == "promotion"
        for item in missing
    )


def _apply_dashboard_contract_attention(
    contracts: list[dict[str, Any]],
    modules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for contract in contracts:
        copy = dict(contract)
        copy["raw_attention"] = bool(contract.get("attention"))
        copy["attention"] = _contract_attention_for_dashboard(contract, modules)
        adjusted.append(copy)
    return adjusted


def _module_name_matches(left: Any, right: Any) -> bool:
    left_name = str(left or "").strip().lower()
    right_name = str(right or "").strip().lower()
    if not left_name or not right_name:
        return False
    if left_name == right_name:
        return True
    aliases = {
        "epis": "epistemic",
        "epistemic": "epis",
    }
    return aliases.get(left_name) == right_name or aliases.get(right_name) == left_name


def _reconcile_contract_progress_from_modules(
    contracts: list[dict[str, Any]],
    modules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use terminal ledgers for work progress while retaining artifact counts."""
    if not modules:
        return contracts
    reconciled: list[dict[str, Any]] = []
    for contract in contracts:
        copy = dict(contract)
        contract_modules: list[dict[str, Any]] = []
        did_reconcile = False
        for contract_module in contract.get("modules") or []:
            if not isinstance(contract_module, dict):
                continue
            module_copy = dict(contract_module)
            module_name = module_copy.get("module")
            output_dir = str(module_copy.get("output_dir") or "")
            ledger_module = next(
                (
                    module
                    for module in modules
                    if _module_name_matches(module_name, module.get("module"))
                    or (
                        output_dir
                        and str(module.get("module_path") or "")
                        and output_dir.rstrip("/").endswith(str(module.get("module_path")).rstrip("/"))
                    )
                ),
                None,
            )
            if ledger_module is not None:
                expected_units = _flow_unit_count(module_copy.get("expected_units"))
                ledger_complete_units = _module_completed_units(ledger_module, expected_units)
                artifact_complete_units = _flow_unit_count(module_copy.get("complete_units"))
                if ledger_complete_units != artifact_complete_units:
                    module_copy["artifact_complete_units"] = artifact_complete_units
                    module_copy["complete_units"] = ledger_complete_units
                    did_reconcile = True
            contract_modules.append(module_copy)
        if contract_modules:
            copy["modules"] = contract_modules
            expected_units = sum(_flow_unit_count(item.get("expected_units")) for item in contract_modules)
            complete_units = sum(_flow_unit_count(item.get("complete_units")) for item in contract_modules)
            if complete_units != _flow_unit_count(contract.get("complete_units")):
                copy["artifact_complete_units"] = _flow_unit_count(contract.get("complete_units"))
                copy["complete_units"] = complete_units
                copy["missing_units"] = max(0, expected_units - complete_units)
                copy["progress_percent"] = (
                    round(min(100.0, (complete_units / expected_units) * 100), 1)
                    if expected_units
                    else 0
                )
                did_reconcile = True
        if did_reconcile:
            copy["ledger_progress_reconciled"] = True
        reconciled.append(copy)
    return reconciled


def _matching_plan_module(
    contract: dict[str, Any],
    plans: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    contract_path = str(contract.get("path") or contract.get("contract_path") or "")
    contract_module_names = [
        module.get("module")
        for module in contract.get("modules") or []
        if isinstance(module, dict)
    ]
    for plan in plans:
        for plan_module in plan.get("modules") or []:
            if not isinstance(plan_module, dict):
                continue
            plan_contract_path = str(plan_module.get("contract_path") or "")
            if contract_path and plan_contract_path and contract_path == plan_contract_path:
                return plan, plan_module
            plan_module_name = plan_module.get("module")
            if any(_module_name_matches(name, plan_module_name) for name in contract_module_names):
                return plan, plan_module
    return None, None


def _missing_operator_value(value: Any) -> bool:
    return value in (None, "", [], {})


def _enrich_contracts_from_plans(
    contracts: list[dict[str, Any]],
    plans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill display-only contract metadata from RUN_PLAN when runtime contracts are thin."""
    if not plans:
        return contracts

    enriched: list[dict[str, Any]] = []
    for contract in contracts:
        plan, plan_module = _matching_plan_module(contract, plans)
        if not plan or not plan_module:
            enriched.append(contract)
            continue

        copy = dict(contract)
        reconciled_fields: list[str] = []
        for field in ("model_selector", "judge_set", "execute_command", "score_command"):
            value = plan_module.get(field) or plan.get(field)
            if _missing_operator_value(copy.get(field)) and not _missing_operator_value(value):
                copy[field] = value
                reconciled_fields.append(field)
        if _missing_operator_value(copy.get("expected_units")) and plan_module.get("expected_units"):
            copy["expected_units"] = plan_module.get("expected_units")
            reconciled_fields.append("expected_units")

        modules: list[dict[str, Any]] = []
        for module in copy.get("modules") or []:
            if not isinstance(module, dict):
                modules.append(module)
                continue
            module_copy = dict(module)
            if _module_name_matches(module_copy.get("module"), plan_module.get("module")):
                for field in ("model_selector", "judge_set", "output_dir", "contract_path"):
                    value = plan_module.get(field) or plan.get(field)
                    if _missing_operator_value(module_copy.get(field)) and not _missing_operator_value(value):
                        module_copy[field] = value
                        reconciled_fields.append(f"module.{field}")
                if _missing_operator_value(module_copy.get("expected_units")) and plan_module.get("expected_units"):
                    module_copy["expected_units"] = plan_module.get("expected_units")
                    reconciled_fields.append("module.expected_units")
            modules.append(module_copy)
        if modules:
            copy["modules"] = modules

        if reconciled_fields:
            copy["plan_reconciled"] = True
            copy["plan_reconciled_fields"] = sorted(set(reconciled_fields))
            copy["prepared_plan"] = {
                "path": plan.get("path"),
                "run_id": plan.get("run_id"),
                "module": plan_module.get("module"),
                "output_dir": plan_module.get("output_dir"),
                "expected_units": plan_module.get("expected_units"),
            }
        enriched.append(copy)
    return enriched


def _model_key(model: dict[str, Any]) -> str:
    return str(model.get("key") or model.get("label") or model.get("model_id") or "model")


def _model_condition_summary(contract: dict[str, Any] | None) -> dict[str, Any]:
    if not contract:
        return {"label": "Model set not recorded", "names": [], "count": 0, "selector": None}
    models = [model for model in contract.get("expected_models") or [] if isinstance(model, dict)]
    names = [_model_key(model) for model in models]
    model_ids = [str(model.get("model_id") or "") for model in models]
    endpoints = [str(model.get("endpoint") or "") for model in models]
    count = len(models)
    unique_model_ids = {model_id for model_id in model_ids if model_id}
    count_noun = "condition" if count > len(unique_model_ids) else "model"
    count_label = f"{count} {count_noun}" if count == 1 else f"{count} {count_noun}s"
    openrouter_slug_prefixes = ("openai/", "anthropic/", "google/", "meta-llama/", "mistralai/")
    if count and all(endpoint == "anthropic_native" for endpoint in endpoints):
        condition = "Anthropic native"
    elif count and all(endpoint == "openai_responses" for endpoint in endpoints):
        condition = "OpenAI Responses"
    elif count and (
        any(_is_openrouter_endpoint(endpoint) for endpoint in endpoints)
        or any(model_id.startswith(openrouter_slug_prefixes) for model_id in model_ids)
    ):
        condition = "Raw OpenRouter"
    elif count and any(endpoint and not _is_openrouter_endpoint(endpoint) for endpoint in endpoints):
        condition = "OpenAI-compatible endpoint"
    elif count and any("/" in model_id for model_id in model_ids):
        condition = "Provider-routed"
    else:
        condition = "Model set"
    return {
        "label": f"{condition} · {count_label}",
        "names": names,
        "count": count,
        "selector": contract.get("model_selector"),
    }


def _sample_summary(contract: dict[str, Any] | None, contract_module: dict[str, Any] | None) -> str:
    identity = (contract or {}).get("identity") or {}
    sample_spec = identity.get("sample_spec") if isinstance(identity.get("sample_spec"), dict) else {}
    scenarios = sample_spec.get("scenario_ids") or (contract_module or {}).get("scenarios") or []
    if scenarios:
        runs = sample_spec.get("runs") or (contract_module or {}).get("runs")
        run_label = f" · r{runs}" if runs else ""
        return f"{', '.join(str(scenario) for scenario in scenarios)}{run_label}"
    compact_case_count = _flow_unit_count(sample_spec.get("case_count"))
    if compact_case_count:
        compact_conversation_count = _flow_unit_count(sample_spec.get("conversation_count"))
        compact_test_type_count = _flow_unit_count(sample_spec.get("test_type_count"))
        epistemic = _normalized_benchmark_module((contract_module or {}).get("module")) == "epistemic"
        parts = []
        if epistemic and compact_conversation_count:
            parts.append(
                f"n={compact_conversation_count} conversation"
                if compact_conversation_count == 1
                else f"n={compact_conversation_count} conversations"
            )
        if compact_case_count:
            case_label = "scored case" if epistemic else "case"
            parts.append(f"{compact_case_count} {case_label}" if compact_case_count == 1 else f"{compact_case_count} {case_label}s")
        if compact_conversation_count and not epistemic:
            parts.append(
                f"{compact_conversation_count} conversation"
                if compact_conversation_count == 1
                else f"{compact_conversation_count} conversations"
            )
        if compact_test_type_count:
            parts.append(
                f"{compact_test_type_count} test type"
                if compact_test_type_count == 1
                else f"{compact_test_type_count} test types"
            )
        return " · ".join(parts)
    sample_items = sample_spec.get("items")
    if isinstance(sample_items, dict):
        case_count = sum(
            len(items)
            for items in sample_items.values()
            if isinstance(items, list)
        )
        conversation_count = _flow_unit_count((contract_module or {}).get("expected_units"))
        test_types = sample_spec.get("test_types")
        test_type_count = len(test_types) if isinstance(test_types, list) else len(sample_items)
        epistemic = _normalized_benchmark_module((contract_module or {}).get("module")) == "epistemic"
        parts = []
        if epistemic and conversation_count:
            parts.append(
                f"n={conversation_count} conversation"
                if conversation_count == 1
                else f"n={conversation_count} conversations"
            )
        if case_count:
            case_label = "scored case" if epistemic else "case"
            parts.append(f"{case_count} {case_label}" if case_count == 1 else f"{case_count} {case_label}s")
        if conversation_count and not epistemic:
            parts.append(
                f"{conversation_count} conversation"
                if conversation_count == 1
                else f"{conversation_count} conversations"
            )
        if test_type_count:
            parts.append(
                f"{test_type_count} test type"
                if test_type_count == 1
                else f"{test_type_count} test types"
            )
        if parts:
            return " · ".join(parts)
    item_indices = sample_spec.get("item_indices") or sample_items or []
    if item_indices:
        return f"{len(item_indices)} item" if len(item_indices) == 1 else f"{len(item_indices)} items"
    raw_item_count = sample_spec.get("item_count")
    item_count = (
        raw_item_count
        if isinstance(raw_item_count, int) and not isinstance(raw_item_count, bool)
        else 0
    )
    if item_count:
        return f"{item_count} item" if item_count == 1 else f"{item_count} items"
    expected_units = (contract_module or {}).get("expected_units") or (contract or {}).get("expected_units")
    if expected_units:
        return f"{expected_units} expected units"
    return "sample not recorded"


def _judge_summary(contract: dict[str, Any] | None) -> str:
    if not contract:
        return "judge not recorded"
    judge_set = contract.get("judge_set")
    judges = [judge for judge in contract.get("expected_judges") or [] if isinstance(judge, dict)]
    primary_count = sum(1 for judge in judges if judge.get("role") == "primary")
    panel_count = sum(1 for judge in judges if judge.get("role") == "panel")
    analyzer_count = sum(1 for judge in judges if judge.get("role") == "analyzer")
    parts = []
    if judge_set:
        parts.append(str(judge_set))
    if primary_count:
        parts.append("primary judge" if primary_count == 1 else f"{primary_count} primary judges")
    if analyzer_count:
        parts.append("analyzer")
    if panel_count:
        parts.append(f"{panel_count}-judge panel")
    return " · ".join(parts) if parts else "judge not recorded"


def _flow_title(module_name: Any, sample_summary: str) -> str:
    module = str(module_name or "module").upper()
    if sample_summary and sample_summary != "sample not recorded":
        return f"{module} · {sample_summary}"
    return module


def _matching_contract(
    module: dict[str, Any],
    contracts: list[dict[str, Any]],
    used_indexes: set[int],
) -> tuple[int | None, dict[str, Any] | None, dict[str, Any] | None]:
    module_name = str(module.get("module") or "")
    module_path = str(module.get("module_path") or "")
    for index, contract in enumerate(contracts):
        if index in used_indexes:
            continue
        for contract_module in contract.get("modules") or []:
            contract_name = str(contract_module.get("module") or "")
            output_dir = str(contract_module.get("output_dir") or "")
            if contract_name == module_name or output_dir.endswith(module_path):
                return index, contract, contract_module
    return None, None, None


def _annotate_contract_membership(group: dict[str, Any]) -> None:
    """Mark ledgers outside a group's contracts as supplemental diagnostics."""
    modules = list(group.get("modules") or [])
    contracts = list(group.get("contracts") or [])
    if not contracts:
        for module in modules:
            module["contract_membership"] = "legacy"
        group["supplemental_module_count"] = 0
        return

    used_indexes: set[int] = set()
    supplemental_count = 0
    for module in modules:
        index, _contract, _contract_module = _matching_contract(
            module,
            contracts,
            used_indexes,
        )
        if index is None:
            module["contract_membership"] = "supplemental"
            supplemental_count += 1
            continue
        used_indexes.add(index)
        module["contract_membership"] = "contract"
    group["supplemental_module_count"] = supplemental_count


def _flow_item_from_module(
    *,
    group: dict[str, Any],
    module: dict[str, Any],
    contract: dict[str, Any] | None,
    contract_module: dict[str, Any] | None,
    scheduler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lane_id = _flow_lane_for_module(module)
    progress = module.get("progress") or {}
    percent = progress.get("percent")
    cost = module.get("cost") or {}
    provenance = (contract or {}).get("provenance") or {}
    model_condition = _model_condition_summary(contract)
    sample = _sample_summary(contract, contract_module)
    transcript = module.get("latest_transcript") or {}
    expected_units = (
        progress.get("expected_units")
        or (contract_module or {}).get("expected_units")
    )
    complete_units = (
        progress.get("completed_units")
        or (contract_module or {}).get("complete_units")
    )
    expected_unit_count = _flow_unit_count(expected_units)
    if not _flow_unit_count(complete_units):
        complete_units = _module_completed_units(module, expected_unit_count)
    scheduler_progress = (scheduler or {}).get("progress") or {}
    if scheduler and scheduler.get("state") == "scoring":
        lane_id = "scoring"
    if lane_id == "score_ready" and expected_units and not complete_units:
        complete_units = expected_units
    scheduler_eta = scheduler_progress.get("eta_seconds")
    return {
        "lane": lane_id,
        "title": _flow_title(module.get("module"), sample),
        "run_id": group.get("run_id"),
        "module": module.get("module"),
        "module_path": module.get("module_path"),
        "stage": module.get("stage"),
        "status": module.get("status"),
        "validity": module.get("validity"),
        "severity": module.get("severity"),
        "progress_percent": percent if percent is not None else (100 if lane_id == "score_ready" else 0),
        "turn_saved": progress.get("turn_saved") or 0,
        "planned_turns": progress.get("planned_turns"),
        "scores_saved": progress.get("scores_saved") or 0,
        "expected_units": expected_units,
        "complete_units": complete_units,
        "scheduler_state": (scheduler or {}).get("state"),
        "scheduler_eta_seconds": scheduler_eta,
        "scheduler_eta_basis": scheduler_progress.get("eta_basis"),
        "scheduler_active_units": scheduler_progress.get("active_units"),
        "scheduler_effective_parallelism": scheduler_progress.get("effective_parallelism"),
        "scheduler_average_completed_unit_seconds": scheduler_progress.get("average_completed_unit_seconds"),
        "max_active_calls": ((scheduler or {}).get("settings") or {}).get("max_active_calls"),
        "elapsed": module.get("elapsed"),
        "cost_total_usd": cost.get("total_cost_usd"),
        "updated_at": module.get("updated_at"),
        "latest_event": module.get("latest_event"),
        "latest_transcript": transcript,
        "latest_model_response": transcript.get("model_response"),
        "latest_user_message": transcript.get("user_message"),
        "status_path": module.get("status_path"),
        "output_dir": module.get("output_dir"),
        "disposition": module.get("disposition") or {},
        "analysis_state": module.get("analysis_state") or "candidate",
        "sample_summary": sample,
        "model_summary": model_condition["label"],
        "model_names": model_condition["names"],
        "model_selector": model_condition["selector"],
        "judge_summary": _judge_summary(contract),
        "next_action": _flow_next_action(lane_id, module),
        "execute_command": (contract or {}).get("scheduler_command")
        or _scheduler_run_command(
            (contract or {}).get("path"),
            ((scheduler or {}).get("settings") or {}).get("max_active_calls"),
        ),
        "contract_path": (contract or {}).get("path"),
        "benchmark_condition_hash": provenance.get("benchmark_condition_hash"),
        "sample_condition_hash": provenance.get("sample_condition_hash"),
        "comparison_spec_hash": provenance.get("comparison_spec_hash"),
        "model_conditions_hash": provenance.get("model_conditions_hash"),
        "batch_condition_hash": provenance.get("batch_condition_hash"),
    }


def _flow_item_from_contract(
    *,
    group: dict[str, Any],
    contract: dict[str, Any],
    contract_module: dict[str, Any],
    scheduler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    control = contract.get("control") or {}
    scheduler_state = str((scheduler or {}).get("state") or "")
    if scheduler_state in {"queued", "dry_run"}:
        lane_id = "queued"
    elif scheduler_state == "running":
        lane_id = "generating"
    elif scheduler_state in {"needs_scoring", "scoring", "score_ready", "attention"}:
        lane_id = scheduler_state
    elif scheduler_state == "stopped":
        lane_id = "attention"
    else:
        lane_id = "attention" if contract.get("attention") or control.get("active") else "prepared"
    provenance = contract.get("provenance") or {}
    model_condition = _model_condition_summary(contract)
    sample = _sample_summary(contract, contract_module)
    scheduler_progress = (scheduler or {}).get("progress") or {}
    progress_percent = scheduler_progress.get("percent")
    if progress_percent is None:
        progress_percent = contract.get("progress_percent") or 0
    expected_units = (
        scheduler_progress.get("expected_units")
        or contract_module.get("expected_units")
        or contract.get("expected_units")
        or 0
    )
    complete_units = (
        scheduler_progress.get("completed_units")
        or contract_module.get("complete_units")
        or 0
    )
    status_label = (
        scheduler_state
        or (control.get("label") if control.get("active") else contract.get("lifecycle_state") or "prepared")
    )
    return {
        "lane": lane_id,
        "title": _flow_title(contract_module.get("module"), sample),
        "run_id": group.get("run_id"),
        "module": contract_module.get("module"),
        "module_path": contract_module.get("module") or contract.get("run_id"),
        "stage": contract_module.get("stage") or contract.get("lifecycle_state") or "prepared",
        "status": status_label,
        "validity": "not_score_ready",
        "severity": "attention" if lane_id == "attention" else "idle",
        "progress_percent": progress_percent,
        "expected_units": expected_units,
        "complete_units": complete_units,
        "scheduler_state": scheduler_state or None,
        "scheduler_eta_seconds": scheduler_progress.get("eta_seconds"),
        "scheduler_eta_basis": scheduler_progress.get("eta_basis"),
        "scheduler_active_units": scheduler_progress.get("active_units"),
        "scheduler_effective_parallelism": scheduler_progress.get("effective_parallelism"),
        "scheduler_average_completed_unit_seconds": scheduler_progress.get("average_completed_unit_seconds"),
        "max_active_calls": ((scheduler or {}).get("settings") or {}).get("max_active_calls"),
        "elapsed": "not started",
        "updated_at": (scheduler or {}).get("updated_at") or contract.get("created_at"),
        "latest_event": None,
        "latest_transcript": {},
        "latest_model_response": None,
        "latest_user_message": None,
        "status_path": None,
        "output_dir": None,
        "disposition": {},
        "analysis_state": "candidate",
        "sample_summary": sample,
        "model_summary": model_condition["label"],
        "model_names": model_condition["names"],
        "model_selector": model_condition["selector"],
        "judge_summary": _judge_summary(contract),
        "next_action": _flow_next_action(lane_id),
        "execute_command": contract.get("scheduler_command")
        or _scheduler_run_command(
            contract.get("path"),
            ((scheduler or {}).get("settings") or {}).get("max_active_calls"),
        ),
        "contract_path": contract.get("path"),
        "benchmark_condition_hash": provenance.get("benchmark_condition_hash"),
        "sample_condition_hash": provenance.get("sample_condition_hash"),
        "comparison_spec_hash": provenance.get("comparison_spec_hash"),
        "model_conditions_hash": provenance.get("model_conditions_hash"),
        "batch_condition_hash": provenance.get("batch_condition_hash"),
    }


def _contract_path_key(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return path.resolve().as_posix()
    except OSError:
        return path.as_posix()


def _scheduler_for_contract(
    contract: dict[str, Any] | None,
    schedulers_by_contract: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    return schedulers_by_contract.get(_contract_path_key((contract or {}).get("path")))


def _flow_item_from_scheduler_only(scheduler: dict[str, Any]) -> dict[str, Any]:
    state = str(scheduler.get("state") or "")
    if state in {"queued", "dry_run"}:
        lane_id = "queued"
    elif state == "running":
        lane_id = "generating"
    elif state in {"needs_scoring", "scoring", "score_ready", "attention"}:
        lane_id = state
    elif state == "stopped":
        lane_id = "attention"
    else:
        lane_id = "prepared"
    progress = scheduler.get("progress") or {}
    return {
        "lane": lane_id,
        "title": str(scheduler.get("run_id") or "scheduled run"),
        "run_id": scheduler.get("run_id"),
        "module": None,
        "module_path": _display_path(scheduler.get("state_dir")),
        "stage": "scheduler",
        "status": state or "unknown",
        "validity": "not_score_ready",
        "severity": "attention" if lane_id == "attention" else "idle",
        "progress_percent": progress.get("percent") or 0,
        "expected_units": progress.get("expected_units") or scheduler.get("expected_units") or 0,
        "complete_units": progress.get("completed_units") or scheduler.get("complete_units") or 0,
        "scheduler_state": state,
        "scheduler_eta_seconds": progress.get("eta_seconds"),
        "scheduler_eta_basis": progress.get("eta_basis"),
        "scheduler_active_units": progress.get("active_units"),
        "scheduler_effective_parallelism": progress.get("effective_parallelism"),
        "scheduler_average_completed_unit_seconds": progress.get("average_completed_unit_seconds"),
        "max_active_calls": (scheduler.get("settings") or {}).get("max_active_calls"),
        "elapsed": None,
        "updated_at": scheduler.get("updated_at"),
        "latest_event": progress.get("latest_event"),
        "latest_transcript": {},
        "latest_model_response": None,
        "latest_user_message": None,
        "sample_summary": "",
        "model_summary": "scheduled contract",
        "model_names": [],
        "model_selector": None,
        "judge_summary": "scheduler",
        "next_action": scheduler.get("reason") or _flow_next_action(lane_id),
        "execute_command": scheduler.get("command"),
        "contract_path": _display_path(scheduler.get("contract_path")),
    }


def _flow_unit_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _flow_number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _queue_model_identity(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return text.lower().replace("therapeutic-harness/", "")


def _queue_model_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown model"
    if text.startswith("therapeutic-harness/"):
        return text.removeprefix("therapeutic-harness/")
    return text


def _expected_score_unit_count(units: list[dict[str, Any]]) -> int:
    return len({
        str(unit.get("expected_score_path"))
        for unit in units
        if isinstance(unit, dict) and unit.get("expected_score_path")
    })


def _normalized_benchmark_module(value: Any) -> str:
    module = str(value or "").strip().lower().replace("-", "_")
    return "epistemic" if module in {"epis", "epistemic_sycophancy"} else module


def _score_unit_projection(contract_module: dict[str, Any]) -> dict[str, Any]:
    units = [
        unit
        for unit in contract_module.get("expected_units") or []
        if isinstance(unit, dict)
    ]
    module = _normalized_benchmark_module(contract_module.get("module"))
    explicit = _expected_score_unit_count(units)
    if explicit:
        expected = explicit
        basis = "explicit_score_paths"
    elif module == "sus":
        expected = len(units)
        basis = "sus_run_fallback"
    else:
        expected = 0
        basis = "not_declared"
    generation_label = "runs" if module == "sus" else "conversations"
    score_label = {
        "sus": "runs",
        "aita": "pairs",
        "epistemic": "cases",
    }.get(module, "result bundles")
    return {
        "expected_score_units": expected,
        "score_expected_units": expected,
        "score_unit_basis": basis,
        "generation_unit_label": generation_label,
        "score_unit_label": score_label,
    }


def _scoring_judge_call_count(
    call_plan: Any,
    *,
    module_name: Any = None,
    module_count: int = 1,
) -> int:
    if not isinstance(call_plan, dict):
        return 0
    lines = [line for line in call_plan.get("lines") or [] if isinstance(line, dict)]
    candidates = [
        line
        for line in lines
        if line.get("stage") == "scoring" and line.get("role") == "judge"
    ]
    module = _normalized_benchmark_module(module_name)
    if module and module_count > 1:
        aliases = {module}
        if module == "epistemic":
            aliases.add("epis")
        candidates = [
            line
            for line in candidates
            if any(alias in str(line.get("operation") or "").lower() for alias in aliases)
        ]
    return sum(
        _flow_unit_count((line.get("calls") or {}).get("expected"))
        for line in candidates
    )


def _expected_model_unit_breakdown(contract_module: dict[str, Any]) -> list[dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for unit in contract_module.get("expected_units") or []:
        if not isinstance(unit, dict):
            continue
        raw_id = unit.get("model_key") or unit.get("model") or unit.get("model_id")
        key = _queue_model_identity(raw_id)
        label = unit.get("model_key") or unit.get("model") or unit.get("model_id")
        row = models.setdefault(
            key,
            {
                "id": key,
                "label": _queue_model_label(label),
                "model_id": unit.get("model_id") or unit.get("model"),
                "expected_units": 0,
                "expected_score_units": 0,
                "_score_paths": set(),
            },
        )
        row["expected_units"] += 1
        if unit.get("expected_score_path"):
            row["_score_paths"].add(str(unit["expected_score_path"]))
    for row in models.values():
        row["expected_score_units"] = len(row.pop("_score_paths"))
        if not row["expected_score_units"] and _normalized_benchmark_module(contract_module.get("module")) == "sus":
            row["expected_score_units"] = row["expected_units"]
    return sorted(models.values(), key=lambda item: (str(item.get("label") or ""), str(item.get("id") or "")))


GENERATION_STARTED_EVENTS = {"conversation_started", "run_started", "sus_run_started"}
GENERATION_COMPLETED_EVENTS = {"conversation_completed", "run_completed", "sus_run_completed"}
GENERATION_TERMINAL_EVENTS = GENERATION_COMPLETED_EVENTS | {
    "conversation_failed",
    "conversation_incomplete",
    "run_failed",
    "sus_run_failed",
}
# Re-export from progress_dedupe so external consumers get the canonical set.
SCORING_COMPLETED_EVENTS = _DEDUPE_SCORING_COMPLETED_EVENTS
# Events whose unit keys count toward the "generated" bucket (completed + reused + terminal signals).
_GENERATED_BUCKET_EVENTS = (
    _DEDUPE_COMPLETED_EVENTS | _DEDUPE_REUSED_EVENTS | _DEDUPE_TERMINAL_SIGNAL_EVENTS
)
PROGRESS_EVENT_NAMES = (
    GENERATION_STARTED_EVENTS
    | GENERATION_TERMINAL_EVENTS
    | GENERATION_COMPLETED_EVENTS
    | SCORING_COMPLETED_EVENTS
    | _DEDUPE_REUSED_EVENTS
    | _DEDUPE_TERMINAL_SIGNAL_EVENTS
    | {
        "final_results_saved",
        "score_failed",
        "stage_failed",
        "turn_saved",
        "paid_call_completed",
    }
)
EVIDENCE_LEDGER_EVENTS = {
    "attempt_failure_classified",
    "conversation_started",
    "conversation_completed",
    "conversation_incomplete",
    "stage_started",
    "stage_completed",
    "stage_failed",
    "judge_result_parsed",
    "score_saved",
    "score_reused",
    "result_saved",
    "score_skipped",
    "score_failed",
    "final_results_saved",
}


# Canonical unit-identity key — prefers unit_id over fallback tuple.
# Re-exported from progress_dedupe; kept here so internal callers are unchanged.
_event_unit_key = _dedupe_event_unit_key


def _event_model_progress(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    generated: dict[str, set[tuple[Any, ...]]] = {}
    scored: dict[str, set[tuple[Any, ...]]] = {}
    started: dict[str, set[tuple[Any, ...]]] = {}
    terminal: dict[str, set[tuple[Any, ...]]] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        raw_model = event.get("model_key") or event.get("model") or event.get("model_id")
        if not raw_model:
            continue
        key = _queue_model_identity(raw_model)
        rows.setdefault(
            key,
            {
                "id": key,
                "label": _queue_model_label(event.get("model_key") or event.get("model") or event.get("model_id")),
                "model_id": event.get("model_id") or event.get("model"),
            },
        )
        unit_key = _event_unit_key(event)
        name = str(event.get("event") or "")
        if name in GENERATION_STARTED_EVENTS:
            started.setdefault(key, set()).add(unit_key)
        if name in GENERATION_TERMINAL_EVENTS:
            terminal.setdefault(key, set()).add(unit_key)
        if name in _GENERATED_BUCKET_EVENTS:
            generated.setdefault(key, set()).add(unit_key)
        if name in SCORING_COMPLETED_EVENTS:
            scored.setdefault(key, set()).add(unit_key)

    progress: dict[str, dict[str, Any]] = {}
    for key, row in rows.items():
        active_units = max(0, len(started.get(key, set()) - terminal.get(key, set())))
        progress[key] = {
            **row,
            "completed_units": len(generated.get(key, set())),
            "scored_units": len(scored.get(key, set())),
            "active_units": active_units,
        }
    return progress


def _merge_operational_models(
    expected_models: list[dict[str, Any]],
    event_models: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    represented_model_ids = {
        str(model.get("model_id"))
        for model in expected_models
        if model.get("model_id")
    }
    for model in expected_models:
        key = _queue_model_identity(model.get("id") or model.get("label") or model.get("model_id"))
        rows[key] = {
            "id": key,
            "label": model.get("label") or _queue_model_label(model.get("model_id") or key),
            "model_id": model.get("model_id"),
            "expected_units": _flow_unit_count(model.get("expected_units")),
            "expected_score_units": _flow_unit_count(model.get("expected_score_units")),
            "completed_units": 0,
            "scored_units": 0,
            "active_units": 0,
        }
    for key, model in event_models.items():
        if key not in rows and str(model.get("model_id") or "") in represented_model_ids:
            continue
        row = rows.setdefault(
            key,
            {
                "id": key,
                "label": model.get("label") or _queue_model_label(key),
                "model_id": model.get("model_id"),
                "expected_units": 0,
                "expected_score_units": 0,
                "completed_units": 0,
                "scored_units": 0,
                "active_units": 0,
            },
        )
        if model.get("model_id") and not row.get("model_id"):
            row["model_id"] = model.get("model_id")
        row["completed_units"] = max(_flow_unit_count(row.get("completed_units")), _flow_unit_count(model.get("completed_units")))
        row["scored_units"] = max(_flow_unit_count(row.get("scored_units")), _flow_unit_count(model.get("scored_units")))
        row["active_units"] = max(_flow_unit_count(row.get("active_units")), _flow_unit_count(model.get("active_units")))
        if row["expected_units"]:
            row["completed_units"] = min(row["expected_units"], row["completed_units"])
        if row["expected_score_units"]:
            row["scored_units"] = min(row["expected_score_units"], row["scored_units"])
    return sorted(
        rows.values(),
        key=lambda item: (
            -_flow_unit_count(item.get("active_units")),
            -_flow_unit_count(item.get("completed_units")),
            str(item.get("label") or ""),
        ),
    )


def _flow_group_key(item: dict[str, Any]) -> str:
    """Return the dashboard's semantic lane grouping key.

    A work group is the benchmark/sample shape operators scan for first
    (for example, "SUS · bridge_heights · r3" or "AITA · 20 items"), not every
    historical output directory. Individual cards still keep their run/model
    identity below that rollup.
    """
    return str(item.get("title") or item.get("module") or item.get("run_id") or "run")


def _flow_work_group_count(items: list[dict[str, Any]]) -> int:
    return len({_flow_group_key(item) for item in items})


def _build_flow_lanes(groups: list[dict[str, Any]], schedulers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    lane_items: dict[str, list[dict[str, Any]]] = {lane_id: [] for lane_id, _, _ in FLOW_LANES}
    schedulers = list(schedulers or [])
    schedulers_by_contract = {
        _contract_path_key(scheduler.get("contract_path")): scheduler
        for scheduler in schedulers
        if scheduler.get("contract_path")
    }
    used_scheduler_keys: set[str] = set()
    for group in groups:
        contracts = list(group.get("contracts") or [])
        used_contract_indexes: set[int] = set()
        for module in group.get("modules") or []:
            if module.get("contract_membership") == "supplemental":
                continue
            index, contract, contract_module = _matching_contract(module, contracts, used_contract_indexes)
            if index is not None:
                used_contract_indexes.add(index)
            scheduler = _scheduler_for_contract(contract, schedulers_by_contract)
            if scheduler:
                used_scheduler_keys.add(_contract_path_key(scheduler.get("contract_path")))
            item = _flow_item_from_module(
                group=group,
                module=module,
                contract=contract,
                contract_module=contract_module,
                scheduler=scheduler,
            )
            lane_items.setdefault(item["lane"], []).append(item)
        for index, contract in enumerate(contracts):
            if index in used_contract_indexes:
                continue
            scheduler = _scheduler_for_contract(contract, schedulers_by_contract)
            if scheduler:
                used_scheduler_keys.add(_contract_path_key(scheduler.get("contract_path")))
            contract_modules = contract.get("modules") or [{"module": contract.get("run_id"), "stage": "prepared"}]
            for contract_module in contract_modules:
                item = _flow_item_from_contract(
                    group=group,
                    contract=contract,
                    contract_module=contract_module,
                    scheduler=scheduler,
                )
                lane_items.setdefault(item["lane"], []).append(item)
    for scheduler in schedulers:
        key = _contract_path_key(scheduler.get("contract_path"))
        if key and key in used_scheduler_keys:
            continue
        item = _flow_item_from_scheduler_only(scheduler)
        lane_items.setdefault(item["lane"], []).append(item)

    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        active_rank = 0 if item.get("lane") in {"generating", "scoring", "attention"} else 1
        return active_rank, str(item.get("updated_at") or "")

    lanes = []
    for lane_id, title, description in FLOW_LANES:
        items = sorted(lane_items.get(lane_id, []), key=sort_key, reverse=True)
        complete_units = sum(_flow_unit_count(item.get("complete_units")) for item in items)
        expected_units = sum(_flow_unit_count(item.get("expected_units")) for item in items)
        unit_count = complete_units if complete_units else expected_units
        lanes.append(
            {
                "id": lane_id,
                "title": title,
                "description": description,
                "count": len(items),
                "work_group_count": _flow_work_group_count(items),
                "group_count": _flow_work_group_count(items),
                "unit_count": unit_count,
                "complete_units": complete_units,
                "expected_units": expected_units,
                "items": items,
            }
        )
    return {
        "lanes": lanes,
        "counts": {lane["id"]: lane["count"] for lane in lanes},
        "work_group_counts": {lane["id"]: lane["work_group_count"] for lane in lanes},
        "group_counts": {lane["id"]: lane["work_group_count"] for lane in lanes},
        "unit_counts": {lane["id"]: lane["unit_count"] for lane in lanes},
    }


OPERATIONAL_QUEUE_STAGES = [
    ("prepared", "Prepared", "Command ready; no paid artifact writes yet."),
    ("queued", "Queued", "Accepted by scheduler; waiting for runner capacity."),
    ("generating", "Generating", "Model-under-test calls are writing transcripts."),
    ("needs_scoring", "Needs Scoring", "Generation is clean; judge work is waiting."),
    ("scoring", "Scoring", "Judge or aggregation work is active."),
    ("score_ready", "Scored", "Score artifacts are ready for review."),
    ("attention", "Attention", "Interrupted, failed, stale, or incomplete work."),
]


def _operational_stage_for_item(
    *,
    module: dict[str, Any] | None,
    contract: dict[str, Any] | None,
    scheduler: dict[str, Any] | None,
) -> str:
    scheduler_state = str((scheduler or {}).get("state") or "")
    runner = (scheduler or {}).get("runner") if isinstance((scheduler or {}).get("runner"), dict) else {}
    status = str((module or runner or {}).get("status") or "")
    stage = str((module or runner or {}).get("stage") or "")
    validity = str((module or runner or {}).get("validity") or "")
    if module and _is_rejected_from_analysis(module.get("disposition") or {}):
        return "attention"
    if scheduler_state in {"queued", "dry_run"}:
        return "queued"
    if scheduler_state == "running":
        return "scoring" if stage == "scoring" else "generating"
    if scheduler_state == "scoring":
        return "scoring"
    if scheduler_state in {"attention", "stopped"}:
        return "attention"
    if module and (module.get("severity") == "attention" or status.startswith("failed") or module.get("attention")):
        return "attention"
    if status == "running":
        return "scoring" if stage == "scoring" else "generating"
    if status == "completed" and validity == "score_ready":
        return "score_ready"
    if status == "completed" and validity == "not_score_ready" and stage == "generation":
        return "needs_scoring"
    if contract:
        return "prepared"
    return "generating"


def _operational_attention_units(module: dict[str, Any] | None, active_units: int, remaining_units: int) -> int:
    if active_units:
        return active_units
    attention = (module or {}).get("attention") if isinstance((module or {}).get("attention"), dict) else {}
    incomplete_count = _flow_unit_count(attention.get("incomplete_count"))
    if incomplete_count:
        return incomplete_count
    progress = (module or {}).get("progress") if isinstance((module or {}).get("progress"), dict) else {}
    failures = _flow_unit_count(progress.get("failures"))
    if failures:
        return failures
    return 1 if remaining_units else 0


def _operational_units_for_stage(
    stage_id: str,
    *,
    expected_units: int,
    completed_units: int,
    active_units: int,
    remaining_units: int,
    attention_units: int,
) -> int:
    if stage_id == "prepared":
        return remaining_units or expected_units
    if stage_id == "queued":
        return remaining_units or max(0, expected_units - completed_units) or expected_units
    if stage_id in {"generating", "scoring"}:
        return completed_units or active_units
    if stage_id == "needs_scoring":
        return completed_units or expected_units
    if stage_id == "score_ready":
        return expected_units or completed_units
    if stage_id == "attention":
        return attention_units
    return completed_units or active_units or expected_units


def _module_completed_units(module: dict[str, Any] | None, expected_units: int) -> int:
    progress = (module or {}).get("progress") if isinstance((module or {}).get("progress"), dict) else {}
    completed = max(
        _flow_unit_count(progress.get("conversations_completed")),
        _flow_unit_count(progress.get("scores_saved")),
        _flow_unit_count(progress.get("final_results_saved")),
    )
    status = str((module or {}).get("status") or "")
    validity = str((module or {}).get("validity") or "")
    if status == "completed" and expected_units and (validity == "score_ready" or completed == 0):
        return expected_units
    return min(expected_units, completed) if expected_units else completed


def _active_leases_for_operational_item(
    *,
    group: dict[str, Any],
    contract: dict[str, Any] | None,
    scheduler: dict[str, Any] | None,
    module_name: Any,
    active_leases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return live paid-call leases attributable to one run/module item."""
    run_ids = {
        str(value)
        for value in (
            group.get("run_id"),
            (contract or {}).get("run_id"),
            (scheduler or {}).get("run_id"),
            (scheduler or {}).get("path_group_id"),
        )
        if value
    }
    matches: list[dict[str, Any]] = []
    for lease in active_leases:
        if not isinstance(lease, dict) or str(lease.get("run_id") or "") not in run_ids:
            continue
        lease_module = lease.get("module")
        if lease_module and not _module_name_matches(module_name, lease_module):
            continue
        matches.append(lease)
    return matches


def _operational_item(
    *,
    group: dict[str, Any],
    module: dict[str, Any] | None,
    contract: dict[str, Any] | None,
    contract_module: dict[str, Any] | None,
    scheduler: dict[str, Any] | None,
    active_leases: list[dict[str, Any]],
) -> dict[str, Any]:
    scheduler_progress = (scheduler or {}).get("progress") if isinstance((scheduler or {}).get("progress"), dict) else {}
    event_models = (module or {}).get("event_model_progress")
    if not isinstance(event_models, dict):
        event_models = {}
    expected_models = (contract_module or {}).get("models")
    if not isinstance(expected_models, list):
        expected_models = []
    models = _merge_operational_models(expected_models, event_models)
    module_name = (module or contract_module or {}).get("module") or (contract or {}).get("run_id")
    item_active_leases = _active_leases_for_operational_item(
        group=group,
        contract=contract,
        scheduler=scheduler,
        module_name=module_name,
        active_leases=active_leases,
    )
    for model in models:
        model["active_units"] = 0
    for lease in item_active_leases:
        lease_model = str(lease.get("model") or "")
        matching_model = next(
            (
                model
                for model in models
                if lease_model
                and lease_model in {str(model.get("id") or ""), str(model.get("model_id") or "")}
            ),
            None,
        )
        if matching_model is not None:
            matching_model["active_units"] += 1
    stage_id = _operational_stage_for_item(module=module, contract=contract, scheduler=scheduler)
    generation_expected_units = _flow_unit_count(
        scheduler_progress.get("expected_units")
        or (contract_module or {}).get("expected_units")
        or (contract or {}).get("expected_units")
    )
    generated_units = _flow_unit_count(scheduler_progress.get("completed_units"))
    if not generated_units:
        generated_units = _module_completed_units(module, generation_expected_units)
    if generation_expected_units:
        generated_units = min(generation_expected_units, generated_units)
    elif generated_units:
        generation_expected_units = generated_units

    expected_score_units = _flow_unit_count(
        (contract_module or {}).get("score_expected_units")
        or (contract_module or {}).get("expected_score_units")
    )
    module_progress = (
        (module or {}).get("progress")
        if isinstance((module or {}).get("progress"), dict)
        else {}
    )
    module_counters = (
        (module or {}).get("counters")
        if isinstance((module or {}).get("counters"), dict)
        else {}
    )
    scored_units = max(
        _flow_unit_count(module_progress.get("scores_saved")),
        _flow_unit_count(module_counters.get("events.score_saved")),
    )
    excluded_score_units = max(
        _flow_unit_count(module_progress.get("scores_skipped")),
        _flow_unit_count(module_counters.get("events.score_skipped")),
    )
    resolved_score_units = scored_units + excluded_score_units
    judge_calls_expected = _flow_unit_count((contract_module or {}).get("judge_calls_expected"))
    judge_calls_completed = _flow_unit_count(module_progress.get("judge_calls_completed"))

    if stage_id in {"needs_scoring", "scoring", "score_ready"}:
        expected_units = expected_score_units
        completed_units = resolved_score_units
        if stage_id == "score_ready" and not expected_units:
            expected_units = resolved_score_units or generated_units
        if stage_id == "score_ready" and expected_units and completed_units == 0:
            completed_units = expected_units
    else:
        expected_units = generation_expected_units
        completed_units = generated_units
    if expected_units:
        completed_units = min(expected_units, completed_units)
    elif completed_units:
        expected_units = completed_units

    if stage_id == "score_ready":
        for model in models:
            if model.get("expected_units"):
                model["completed_units"] = model["expected_units"]
            if model.get("expected_score_units"):
                model["scored_units"] = model["expected_score_units"]
    active_units = len(item_active_leases)
    remaining_units = max(0, expected_units - completed_units) if expected_units else 0
    if stage_id not in {"generating", "scoring", "attention"}:
        active_units = 0
    attention_units = _operational_attention_units(module, active_units, remaining_units) if stage_id == "attention" else 0
    units = _operational_units_for_stage(
        stage_id,
        expected_units=expected_units,
        completed_units=completed_units,
        active_units=active_units,
        remaining_units=remaining_units,
        attention_units=attention_units,
    )
    sample = _sample_summary(contract, contract_module)
    title = _flow_title(module_name, sample)
    return {
        "id": ":".join(
            str(part)
            for part in (
                group.get("run_id"),
                module_name,
                (module or {}).get("module_path") or (contract_module or {}).get("module") or title,
            )
            if part
        ),
        "stage": stage_id,
        "title": title,
        "run_id": group.get("run_id"),
        "module": module_name,
        "module_path": (module or {}).get("module_path") or (contract_module or {}).get("module"),
        "status": (module or {}).get("status") or (scheduler or {}).get("state") or (contract or {}).get("lifecycle_state") or "prepared",
        "validity": (module or {}).get("validity"),
        "scheduler_state": (scheduler or {}).get("state"),
        "generation_expected_units": generation_expected_units,
        "generated_units": generated_units,
        "expected_score_units": expected_score_units,
        "score_expected_units": expected_score_units,
        "scored_units": scored_units,
        "excluded_score_units": excluded_score_units,
        "score_completed_units": resolved_score_units,
        "judge_calls_expected": judge_calls_expected,
        "judge_calls_completed": judge_calls_completed,
        "generation_unit_label": (contract_module or {}).get("generation_unit_label") or "work units",
        "score_unit_label": (contract_module or {}).get("score_unit_label") or "result bundles",
        "score_unit_basis": (contract_module or {}).get("score_unit_basis") or "not_declared",
        "expected_units": expected_units,
        "completed_units": completed_units,
        "remaining_units": remaining_units,
        "active_units": active_units,
        "attention_units": attention_units,
        "units": units,
        "progress_percent": (
            round(min(100.0, (completed_units / expected_units) * 100), 1)
            if expected_units
            else None
        ),
        "eta_seconds": scheduler_progress.get("eta_seconds"),
        "eta_basis": scheduler_progress.get("eta_basis"),
        "average_completed_unit_seconds": scheduler_progress.get("average_completed_unit_seconds"),
        "max_active_calls": ((scheduler or {}).get("settings") or {}).get("max_active_calls"),
        "updated_at": (scheduler or {}).get("updated_at") or (module or {}).get("updated_at") or (contract or {}).get("created_at"),
        "models": models,
        "sample_summary": sample,
        "model_summary": _model_condition_summary(contract)["label"],
        "judge_summary": _judge_summary(contract),
        "next_action": _flow_next_action("attention" if stage_id == "attention" else stage_id, module),
    }


def _build_operational_queue(
    groups: list[dict[str, Any]],
    *,
    schedulers: list[dict[str, Any]] | None = None,
    active_leases: list[dict[str, Any]] | None = None,
    active_lease_count: int = 0,
    lease_max_active: int = 0,
    paid_call_max_active: int = 0,
) -> dict[str, Any]:
    stage_items: dict[str, list[dict[str, Any]]] = {stage_id: [] for stage_id, _, _ in OPERATIONAL_QUEUE_STAGES}
    schedulers = list(schedulers or [])
    active_leases = list(active_leases or [])
    schedulers_by_contract = {
        _contract_path_key(scheduler.get("contract_path")): scheduler
        for scheduler in schedulers
        if scheduler.get("contract_path")
    }
    used_scheduler_keys: set[str] = set()

    for group in groups:
        contracts = list(group.get("contracts") or [])
        used_contract_indexes: set[int] = set()
        for module in group.get("modules") or []:
            if module.get("contract_membership") == "supplemental":
                continue
            index, contract, contract_module = _matching_contract(module, contracts, used_contract_indexes)
            if index is not None:
                used_contract_indexes.add(index)
            scheduler = _scheduler_for_contract(contract, schedulers_by_contract)
            if scheduler:
                used_scheduler_keys.add(_contract_path_key(scheduler.get("contract_path")))
            item = _operational_item(
                group=group,
                module=module,
                contract=contract,
                contract_module=contract_module,
                scheduler=scheduler,
                active_leases=active_leases,
            )
            stage_items.setdefault(item["stage"], []).append(item)
        for index, contract in enumerate(contracts):
            if index in used_contract_indexes:
                continue
            scheduler = _scheduler_for_contract(contract, schedulers_by_contract)
            if scheduler:
                used_scheduler_keys.add(_contract_path_key(scheduler.get("contract_path")))
            for contract_module in contract.get("modules") or [{"module": contract.get("run_id"), "stage": "prepared"}]:
                item = _operational_item(
                    group=group,
                    module=None,
                    contract=contract,
                    contract_module=contract_module,
                    scheduler=scheduler,
                    active_leases=active_leases,
                )
                stage_items.setdefault(item["stage"], []).append(item)

    for scheduler in schedulers:
        key = _contract_path_key(scheduler.get("contract_path"))
        if key and key in used_scheduler_keys:
            continue
        item = _operational_item(
            group={"run_id": scheduler.get("run_id") or scheduler.get("path_group_id") or "scheduled-run"},
            module=None,
            contract=None,
            contract_module={"module": scheduler.get("run_id"), "stage": "scheduler"},
            scheduler=scheduler,
            active_leases=active_leases,
        )
        stage_items.setdefault(item["stage"], []).append(item)

    stages: list[dict[str, Any]] = []
    total_expected = 0
    total_completed = 0
    total_active = 0
    total_attention = 0
    total_generation_expected = 0
    total_generation_completed = 0
    total_score_expected = 0
    total_score_completed = 0
    total_judge_calls_expected = 0
    total_judge_calls_completed = 0
    for stage_id, title, description in OPERATIONAL_QUEUE_STAGES:
        items = sorted(
            stage_items.get(stage_id, []),
            key=lambda item: (str(item.get("updated_at") or ""), _flow_unit_count(item.get("active_units"))),
            reverse=True,
        )
        expected_units = sum(_flow_unit_count(item.get("expected_units")) for item in items)
        completed_units = sum(_flow_unit_count(item.get("completed_units")) for item in items)
        generation_expected_units = sum(
            _flow_unit_count(item.get("generation_expected_units")) for item in items
        )
        generated_units = sum(_flow_unit_count(item.get("generated_units")) for item in items)
        expected_score_units = sum(
            _flow_unit_count(item.get("expected_score_units")) for item in items
        )
        scored_units = sum(_flow_unit_count(item.get("scored_units")) for item in items)
        excluded_score_units = sum(
            _flow_unit_count(item.get("excluded_score_units")) for item in items
        )
        score_completed_units = sum(
            _flow_unit_count(item.get("score_completed_units")) for item in items
        )
        judge_calls_expected = sum(
            _flow_unit_count(item.get("judge_calls_expected")) for item in items
        )
        judge_calls_completed = sum(
            _flow_unit_count(item.get("judge_calls_completed")) for item in items
        )
        active_units = sum(_flow_unit_count(item.get("active_units")) for item in items)
        attention_units = sum(_flow_unit_count(item.get("attention_units")) for item in items)
        units = sum(_flow_unit_count(item.get("units")) for item in items)
        total_expected += expected_units
        total_completed += completed_units
        total_active += active_units
        total_attention += attention_units
        total_generation_expected += generation_expected_units
        total_generation_completed += generated_units
        total_score_expected += expected_score_units
        total_score_completed += score_completed_units
        total_judge_calls_expected += judge_calls_expected
        total_judge_calls_completed += judge_calls_completed
        stages.append(
            {
                "id": stage_id,
                "title": title,
                "description": description,
                "units": units,
                "expected_units": expected_units,
                "completed_units": completed_units,
                "generation_expected_units": generation_expected_units,
                "generated_units": generated_units,
                "expected_score_units": expected_score_units,
                "scored_units": scored_units,
                "excluded_score_units": excluded_score_units,
                "score_completed_units": score_completed_units,
                "judge_calls_expected": judge_calls_expected,
                "judge_calls_completed": judge_calls_completed,
                "active_units": active_units,
                "attention_units": attention_units,
                "count": len(items),
                "group_count": _flow_work_group_count(items),
                "items": items,
            }
        )

    return {
        "stages": stages,
        "total_units": total_generation_expected,
        "generated_units": total_generation_completed,
        "active_units": total_active,
        "attention_units": total_attention,
        "generation_expected_units": total_generation_expected,
        "generation_completed_units": total_generation_completed,
        "score_expected_units": total_score_expected,
        "score_completed_units": total_score_completed,
        "judge_calls_expected": total_judge_calls_expected,
        "judge_calls_completed": total_judge_calls_completed,
        "leases": {
            "active": active_lease_count,
            "cap": paid_call_max_active,
            "registry_cap": lease_max_active,
            "source": "policy",
        },
    }


def _select_latest_group(grouped: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the sticky "latest" run.

    Among groups that are actively running, choose a *stable* member so the
    resolved run does not flip every refresh when several runs write
    concurrently: earliest ``started_at`` wins, tie-broken by ``run_id``
    lexicographically. When nothing is running, most-recently-updated is fine.
    """
    running = [group for group in grouped if group.get("running_count")]
    if running:
        return min(
            running,
            # Missing started_at sorts last via a high sentinel so runs with a
            # real start timestamp are preferred as the sticky anchor.
            key=lambda group: (
                str(group.get("started_at") or "￿"),
                str(group.get("run_id") or ""),
            ),
        )
    return next(
        (
            group
            for group in grouped
            if group.get("updated_at") or group.get("latest_event_at")
        ),
        None,
    )


PREREG_FREEZE_FILENAME = "PREREG_FREEZE.json"
COMPANION_DIRNAME = ".benchmark-companion"
COMPANION_ACTIVE_FILENAME = "ACTIVE.json"
COMPANION_RESUME_FILENAME = "RESUME.json"


def _load_prereg_family_key(root: Path, run_id: str) -> str | None:
    """Return the shared prereg fingerprint for a run dir, if frozen.

    All member dirs of one logical run carry ``PREREG_FREEZE.json`` with the
    same ``prereg_sha256_current`` — that value is the family key.
    """
    if not run_id:
        return None
    freeze = _load_json(root / run_id / PREREG_FREEZE_FILENAME)
    sha = freeze.get("prereg_sha256_current") if isinstance(freeze, dict) else None
    return sha if isinstance(sha, str) and sha else None


def _prereg_families(grouped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build family entries for any prereg fingerprint shared by >= 2 runs."""
    members: dict[str, list[str]] = {}
    for group in grouped:
        sha = group.get("prereg_sha256")
        if isinstance(sha, str) and sha:
            members.setdefault(sha, []).append(str(group.get("run_id") or ""))
    families: list[dict[str, Any]] = []
    for sha, run_ids in members.items():
        run_ids = sorted({run_id for run_id in run_ids if run_id})
        if len(run_ids) < 2:
            continue
        families.append(
            {
                "key": f"family:{sha}",
                "prereg_sha256": sha,
                "short_sha": sha[:12],
                "member_run_ids": run_ids,
                "member_count": len(run_ids),
            }
        )
    families.sort(key=lambda family: family["prereg_sha256"])
    return families


def _active_workflow_scope(
    root: Path,
    grouped: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the active companion workflow as a dashboard-only run scope.

    Companion state supplies membership only. Progress and totals still come
    from the scientific ledgers loaded into ``grouped``.
    """
    state_root: Path | None = None
    for candidate in (root.resolve(), *root.resolve().parents):
        companion = candidate / COMPANION_DIRNAME
        if (companion / COMPANION_ACTIVE_FILENAME).is_file():
            state_root = companion
            break
    if state_root is None:
        return None

    active = _load_json(state_root / COMPANION_ACTIVE_FILENAME)
    workflow_id = str(active.get("workflow_id") or "")
    if not workflow_id or Path(workflow_id).name != workflow_id:
        return None
    workflow_dir = (state_root / workflow_id).resolve()
    try:
        workflow_dir.relative_to(state_root.resolve())
    except ValueError:
        return None
    resume = _load_json(workflow_dir / COMPANION_RESUME_FILENAME)
    if str(resume.get("workflow_id") or "") != workflow_id:
        return None

    available = {str(group.get("run_id") or "") for group in grouped}
    member_run_ids = sorted(
        {
            str(run.get("run_id") or "")
            for run in resume.get("runs") or []
            if isinstance(run, dict)
            and str(run.get("run_id") or "") in available
        }
    )
    if not member_run_ids:
        return None
    return {
        "key": "workflow:active",
        "workflow_id": workflow_id,
        "member_run_ids": member_run_ids,
        "member_count": len(member_run_ids),
    }


@dataclass(frozen=True)
class DashboardOptions:
    results_root: Path
    event_limit: int = 80
    summary_only: bool = False


def build_dashboard_data(options: DashboardOptions) -> dict[str, Any]:
    """Return grouped run status suitable for JSON and the browser UI."""
    root = options.results_root
    now = datetime.now(timezone.utc)
    modules: list[dict[str, Any]] = []
    latest_events: list[dict[str, Any]] = []
    evidence_feed: list[dict[str, Any]] = []
    ledger_warnings: list[dict[str, str]] = []
    contracts = (
        _load_contract_headers(root, warnings=ledger_warnings)
        if options.summary_only
        else _load_contract_summaries(root, warnings=ledger_warnings)
    )
    contracts_by_group_module = {
        (
            str(contract.get("path_group_id") or contract.get("run_id") or ""),
            str(contract_module.get("module") or ""),
        ): contract
        for contract in contracts
        if isinstance(contract, dict)
        for contract_module in contract.get("modules") or []
        if isinstance(contract_module, dict)
    }
    plans = _load_run_plans(root, warnings=ledger_warnings)
    schedulers = _load_scheduler_summaries(root)
    try:
        capacity_report = sanitize_ledger_value(
            paid_call_capacity_report(
                environment=read_repo_env_values(PAID_CALL_LIMIT_ENV_NAMES)
            )
        )
    except ValueError as exc:
        ledger_warnings.append(
            {
                "kind": "paid_call_capacity",
                "path": ".env",
                "error": str(exc),
            }
        )
        capacity_report = sanitize_ledger_value(paid_call_capacity_report(environment={}))
        capacity_report["configuration_error"] = str(exc)
    lease_status = sanitize_ledger_value(load_paid_call_lease_status())
    active_leases = lease_status.get("active_leases")
    if not isinstance(active_leases, list):
        active_leases = []
    rate_limit_cooldowns = lease_status.get("rate_limit_cooldowns")
    if not isinstance(rate_limit_cooldowns, list):
        rate_limit_cooldowns = []
    lease_max_active = capacity_report.get(
        "effective_limit",
        lease_status.get("max_active_calls", configured_max_active_calls()),
    )
    try:
        lease_max_active_count = int(lease_max_active)
    except (TypeError, ValueError):
        lease_max_active_count = 0

    for status_path in _iter_ledger_paths(root, "RUN_STATUS.json"):
        output_dir = status_path.parent
        status = sanitize_ledger_value(_load_json(status_path))
        invalid_field = next(
            (
                field
                for field in ("module", "stage", "status", "validity")
                if status.get(field) is not None and not isinstance(status.get(field), str)
            ),
            None,
        )
        if invalid_field:
            ledger_warnings.append(
                {
                    "kind": "run_status",
                    "path": _relative(status_path, REPO_ROOT),
                    "error": f"{invalid_field} must be a string",
                }
            )
            continue
        disposition = _load_disposition(output_dir)
        rejected_from_analysis = _is_rejected_from_analysis(disposition)
        events_path = output_dir / "RUN_EVENTS.jsonl"
        progress_events = _load_events_filtered(
            events_path,
            names=PROGRESS_EVENT_NAMES | EVIDENCE_LEDGER_EVENTS,
        )
        events = _load_events(events_path, limit=max(options.event_limit, 1))
        rel_parts = output_dir.resolve().relative_to(root.resolve()).parts
        group = rel_parts[0] if rel_parts else output_dir.name
        module_path = "/".join(rel_parts[1:]) if len(rel_parts) > 1 else status.get("module") or output_dir.name
        progress = _event_progress(progress_events, status)
        progress["source"] = "full_ledger_events"
        base_severity = _severity(status.get("status"), status.get("validity"))
        latest_event = _event_digest(events[-1], group=group, module_path=module_path) if events else None
        latest_write_at = status.get("updated_at") or (latest_event or {}).get("timestamp")
        updated_age_seconds = _age_seconds(latest_write_at, now)
        stale_running = (
            base_severity == "running"
            and updated_age_seconds is not None
            and updated_age_seconds > STALE_RUNNING_SECONDS
        )
        severity = "attention" if stale_running else base_severity
        attention = _attention_summary(status, events)
        if stale_running:
            attention = {
                "title": "Stale running ledger",
                "reason": (
                    f"No ledger writes for {_format_duration(updated_age_seconds)}. "
                    "Inspect before spending more model or judge calls."
                ),
                "incomplete_count": 0,
                "incomplete_examples": [],
                "failure_stage": status.get("stage"),
                "latest_failure_event": (latest_event or {}).get("event"),
                "action": "Pause paid collection, inspect the ledger and provider health, then resume or rerun deliberately.",
            }
        if rejected_from_analysis:
            severity = "rejected"
            attention = None
        if latest_event:
            latest_events.append(latest_event)
        for event in events[-10:]:
            latest_events.append(_event_digest(event, group=group, module_path=module_path))
        if not options.summary_only:
            evidence_feed.extend(
                _evidence_items_from_module(
                    group=group,
                    module_path=module_path,
                    status=status,
                    events=progress_events,
                    output_dir=output_dir,
                )
            )
        eligibility = _disposition_eligibility(status, disposition)
        module = {
            "group": group,
            "module": status.get("module") or module_path,
            "module_path": module_path,
            "stage": status.get("stage"),
            "status": status.get("status", "unknown"),
            "validity": status.get("validity", "unknown"),
            "severity": severity,
            "output_dir": _relative(output_dir, REPO_ROOT),
            "status_path": _relative(status_path, REPO_ROOT),
            "events_path": _relative(events_path, REPO_ROOT),
            "started_at": status.get("started_at"),
            "updated_at": status.get("updated_at"),
            "updated_age_seconds": updated_age_seconds,
            "stale_running": stale_running,
            "completed_at": status.get("completed_at"),
            "failed_at": status.get("failed_at"),
            "elapsed_seconds": _elapsed_seconds(status, now),
            "elapsed": _format_duration(_elapsed_seconds(status, now)),
            "metadata": status.get("metadata") if isinstance(status.get("metadata"), dict) else {},
            "counters": status.get("counters") if isinstance(status.get("counters"), dict) else {},
            "cost": _cost_summary(status),
            "spend_guard": _spend_guard(
                status,
                contracts_by_group_module.get(
                    (group, str(status.get("module") or module_path))
                ),
            ),
            "progress": progress,
            "event_model_progress": _event_model_progress(progress_events),
            "score_state": _score_state(
                status,
                (attention or {}).get("classification")
                if isinstance(attention, dict)
                else None,
            ),
            "disposition": disposition,
            "analysis_state": disposition.get("disposition") or "candidate",
            **eligibility,
            "attention": attention,
            "latest_event": latest_event,
            "latest_transcript": (
                None
                if options.summary_only
                else _transcript_preview(events, output_dir, status)
            ),
            "recent_events": events[-20:],
        }
        modules.append(module)

    groups: dict[str, dict[str, Any]] = {}
    severity_rank = {"attention": 3, "running": 2, "ready": 1, "idle": 0, "rejected": 0}
    for module in modules:
        group = groups.setdefault(
            module["group"],
            {
                "run_id": module["group"],
                "modules": [],
                "statuses": {},
                "validity": {},
                "severity": "idle",
                "attention_count": 0,
                "running_count": 0,
                "ready_count": 0,
                "started_at": module.get("started_at"),
                "updated_at": module.get("updated_at"),
                "latest_event_at": None,
                "elapsed_seconds": 0,
                "cost_total_usd": 0.0,
                "contracts": [],
                "plans": [],
                "schedulers": [],
            },
        )
        group["modules"].append(module)
        group["statuses"][module["status"]] = group["statuses"].get(module["status"], 0) + 1
        group["validity"][module["validity"]] = group["validity"].get(module["validity"], 0) + 1
        group[f"{module['severity']}_count"] = group.get(f"{module['severity']}_count", 0) + 1
        if severity_rank[module["severity"]] > severity_rank[group["severity"]]:
            group["severity"] = module["severity"]
        group["started_at"] = min(
            [value for value in (group.get("started_at"), module.get("started_at")) if value],
            default=None,
        )
        group["updated_at"] = max(
            [value for value in (group.get("updated_at"), module.get("updated_at")) if value],
            default=None,
        )
        group["latest_event_at"] = max(
            [value for value in (group.get("latest_event_at"), (module.get("latest_event") or {}).get("timestamp")) if value],
            default=None,
        )
        group["elapsed_seconds"] = max(group["elapsed_seconds"], module.get("elapsed_seconds") or 0)
        if module.get("cost"):
            group["cost_total_usd"] += module["cost"]["total_cost_usd"]

    grouped = sorted(groups.values(), key=lambda item: item.get("updated_at") or "", reverse=True)
    contracts_by_run: dict[str, list[dict[str, Any]]] = {}
    for contract in contracts:
        path_group_id = str(contract.get("path_group_id") or "")
        run_id = path_group_id or str(contract.get("run_id") or "unknown")
        contracts_by_run.setdefault(run_id, []).append(contract)
    plans_by_run: dict[str, list[dict[str, Any]]] = {}
    for plan in plans:
        path_group_id = str(plan.get("path_group_id") or "")
        run_id = path_group_id or str(plan.get("run_id") or "unknown")
        plans_by_run.setdefault(run_id, []).append(plan)
    schedulers_by_run: dict[str, list[dict[str, Any]]] = {}
    for scheduler in schedulers:
        path_group_id = str(scheduler.get("path_group_id") or "")
        run_id = path_group_id or str(scheduler.get("run_id") or "unknown")
        schedulers_by_run.setdefault(run_id, []).append(scheduler)

    for group in grouped:
        group_plans = plans_by_run.pop(group["run_id"], [])
        group_contracts = _apply_dashboard_contract_attention(
            _reconcile_contract_progress_from_modules(
                _enrich_contracts_from_plans(
                    contracts_by_run.pop(group["run_id"], []),
                    group_plans,
                ),
                group["modules"],
            ),
            group["modules"],
        )
        group_schedulers = schedulers_by_run.pop(group["run_id"], [])
        group["contracts"] = group_contracts
        group["plans"] = group_plans
        group["schedulers"] = group_schedulers
        group["contract_attention_count"] = sum(1 for contract in group_contracts if contract.get("attention"))
        group["elapsed"] = _format_duration(group.get("elapsed_seconds"))
        group["cost_total_usd"] = round(group["cost_total_usd"], 4)
        group["modules"] = sorted(
            group["modules"],
            key=lambda item: (str(item.get("module_path")), str(item.get("module"))),
        )
        _annotate_contract_membership(group)
    for run_id, group_contracts in contracts_by_run.items():
        group_plans = plans_by_run.pop(run_id, [])
        group_contracts = _enrich_contracts_from_plans(group_contracts, group_plans)
        group_contracts = _apply_dashboard_contract_attention(group_contracts, [])
        grouped.append(
            {
                "run_id": run_id,
                "modules": [],
                "statuses": {},
                "validity": {},
                "severity": "attention" if any(contract.get("attention") for contract in group_contracts) else "idle",
                "attention_count": 0,
                "running_count": 0,
                "ready_count": 0,
                "started_at": min(
                    [value for value in (contract.get("created_at") for contract in group_contracts) if value],
                    default=None,
                ),
                "updated_at": max(
                    [value for value in (contract.get("created_at") for contract in group_contracts) if value],
                    default=None,
                ),
                "latest_event_at": None,
                "elapsed_seconds": 0,
                "elapsed": "0s",
                "cost_total_usd": 0.0,
                "contracts": group_contracts,
                "plans": group_plans,
                "schedulers": schedulers_by_run.pop(run_id, []),
                "contract_attention_count": sum(1 for contract in group_contracts if contract.get("attention")),
                "contract_only": True,
            }
        )
    for run_id, group_plans in plans_by_run.items():
        group_schedulers = schedulers_by_run.pop(run_id, [])
        grouped.append(
            {
                "run_id": run_id,
                "modules": [],
                "statuses": {},
                "validity": {},
                "severity": "idle",
                "attention_count": 0,
                "running_count": 0,
                "ready_count": 0,
                "started_at": min(
                    [value for value in (plan.get("created_at") for plan in group_plans) if value],
                    default=None,
                ),
                "updated_at": max(
                    [value for value in (plan.get("updated_at") for plan in group_plans) if value],
                    default=None,
                ),
                "latest_event_at": None,
                "elapsed_seconds": 0,
                "elapsed": "0s",
                "cost_total_usd": 0.0,
                "contracts": [],
                "plans": group_plans,
                "schedulers": group_schedulers,
                "contract_attention_count": 0,
                "contract_only": True,
            }
        )
    for run_id, group_schedulers in schedulers_by_run.items():
        grouped.append(
            {
                "run_id": run_id,
                "modules": [],
                "statuses": {},
                "validity": {},
                "severity": "attention" if any(str(item.get("state") or "") in {"attention", "stopped"} for item in group_schedulers) else "idle",
                "attention_count": 0,
                "running_count": 0,
                "ready_count": 0,
                "started_at": min(
                    [value for value in (scheduler.get("started_at") or scheduler.get("created_at") for scheduler in group_schedulers) if value],
                    default=None,
                ),
                "updated_at": max(
                    [value for value in (scheduler.get("updated_at") for scheduler in group_schedulers) if value],
                    default=None,
                ),
                "latest_event_at": None,
                "elapsed_seconds": 0,
                "elapsed": "0s",
                "cost_total_usd": 0.0,
                "contracts": [],
                "plans": [],
                "schedulers": group_schedulers,
                "contract_attention_count": 0,
                "contract_only": True,
            }
        )

    # Re-sort after contract-only/plan-only/scheduler-only groups are appended
    # so a freshly prepared run can become the latest run before its first
    # ledger write.
    grouped.sort(key=lambda item: item.get("updated_at") or "", reverse=True)

    # Tag each group with its shared prereg fingerprint (family key) so the UI
    # can offer a logical-run family view across sibling run dirs.
    for group in grouped:
        group["prereg_sha256"] = _load_prereg_family_key(
            root, str(group.get("run_id") or "")
        )
    families = _prereg_families(grouped)
    active_workflow = _active_workflow_scope(root, grouped)
    workflows = [active_workflow] if active_workflow is not None else []

    latest_events = sorted(
        latest_events,
        key=_event_sort_key,
        reverse=True,
    )
    seen_event_keys: set[tuple[str, str, str, int]] = set()
    deduped_latest_events: list[dict[str, Any]] = []
    for event in latest_events:
        key = (
            str(event.get("group")),
            str(event.get("module_path")),
            str(event.get("event")),
            event.get("sequence") if isinstance(event.get("sequence"), int) else -1,
        )
        if key in seen_event_keys:
            continue
        seen_event_keys.add(key)
        deduped_latest_events.append(event)
        if len(deduped_latest_events) >= 60:
            break

    contracts = [
        contract
        for group in grouped
        for contract in group.get("contracts", [])
        if isinstance(contract, dict)
    ]
    active_analysis_modules = [
        module
        for module in modules
        if not _is_rejected_from_analysis(module.get("disposition") or {})
    ]
    rejected_modules = [
        module
        for module in modules
        if _is_rejected_from_analysis(module.get("disposition") or {})
    ]
    score_ready_count = sum(1 for module in active_analysis_modules if module.get("validity") == "score_ready")
    tracked_cost_total = sum(
        module["cost"]["total_cost_usd"] for module in modules if module.get("cost")
    )
    active_elapsed_seconds = max(
        [module.get("elapsed_seconds") or 0 for module in modules if module.get("severity") == "running"],
        default=0,
    )
    suite_elapsed_seconds = max(
        [group.get("elapsed_seconds") or 0 for group in grouped],
        default=0,
    )
    latest_group = _select_latest_group(grouped)
    latest_elapsed_seconds = (latest_group or {}).get("elapsed_seconds") or 0
    contract_attention_count = sum(1 for contract in contracts if contract.get("attention"))
    active_control_count = sum(
        1 for contract in contracts if (contract.get("control") or {}).get("active")
    )
    scheduler_running_count = sum(1 for scheduler in schedulers if scheduler.get("state") in {"running", "scoring"})
    scheduler_queued_count = sum(1 for scheduler in schedulers if scheduler.get("state") in {"queued", "dry_run"})
    scheduler_attention_count = sum(1 for scheduler in schedulers if scheduler.get("state") in {"attention", "stopped"})
    scheduler_eta_values = [
        scheduler.get("eta_seconds")
        for scheduler in schedulers
        if scheduler.get("state") in {"running", "scoring"} and isinstance(scheduler.get("eta_seconds"), (int, float))
    ]
    displayed_paid_call_max_active = lease_max_active_count
    flow = _build_flow_lanes(grouped, schedulers=schedulers)
    operational_queue = _build_operational_queue(
        grouped,
        schedulers=schedulers,
        active_leases=active_leases,
        active_lease_count=len(active_leases),
        lease_max_active=lease_max_active_count,
        paid_call_max_active=displayed_paid_call_max_active,
    )

    return {
        "schema_version": "benchmark-live-dashboard-v1",
        "generated_at": now.isoformat(),
        "results_root": _relative(root, REPO_ROOT),
        "groups": grouped,
        "families": families,
        "workflows": workflows,
        "module_count": len(modules),
        "contracts": contracts,
        "plans": plans,
        "schedulers": schedulers,
        "paid_call_leases": {
            "lease_dir": _relative(default_lease_dir(), REPO_ROOT),
            **lease_status,
            "max_active_calls": lease_max_active_count,
            "capacity": capacity_report,
            "active_leases": active_leases,
            "rate_limit_cooldowns": rate_limit_cooldowns,
        },
        "flow": flow,
        "operational_queue": operational_queue,
        "summary": {
            "attention_count": sum(1 for module in active_analysis_modules if module.get("severity") == "attention"),
            "running_count": sum(1 for module in active_analysis_modules if module.get("severity") == "running"),
            "ready_count": sum(1 for module in active_analysis_modules if module.get("severity") == "ready"),
            "score_ready_count": score_ready_count,
            "failed_count": sum(1 for module in active_analysis_modules if str(module.get("status") or "").startswith("failed")),
            "rejected_count": len(rejected_modules),
            "tracked_cost_total_usd": round(tracked_cost_total, 4),
            "latest_event_at": deduped_latest_events[0].get("timestamp") if deduped_latest_events else None,
            "active_elapsed": _format_duration(active_elapsed_seconds) if active_elapsed_seconds else "none",
            "latest_elapsed": _format_duration(latest_elapsed_seconds) if latest_elapsed_seconds else "none",
            "latest_run_id": (latest_group or {}).get("run_id"),
            "suite_elapsed": _format_duration(suite_elapsed_seconds) if suite_elapsed_seconds else "none",
            "spend_attention_count": sum(
                1
                for module in modules
                if (module.get("spend_guard") or {}).get("severity") == "attention"
            ),
            "contract_count": len(contracts),
            "plan_count": len(plans),
            "contract_attention_count": contract_attention_count,
            "contract_expected_units": sum(int(contract.get("expected_units") or 0) for contract in contracts),
            "contract_complete_units": sum(int(contract.get("complete_units") or 0) for contract in contracts),
            "scheduler_completed_units": operational_queue.get("generated_units", 0),
            "scheduler_active_units": operational_queue.get("active_units", 0),
            "scheduler_attention_units": operational_queue.get("attention_units", 0),
            "active_control_count": active_control_count,
            "scheduler_count": len(schedulers),
            "scheduler_running_count": scheduler_running_count,
            "scheduler_queued_count": scheduler_queued_count,
            "scheduler_attention_count": scheduler_attention_count,
            "scheduler_eta_seconds": min(scheduler_eta_values) if scheduler_eta_values else None,
            "paid_call_active_count": len(active_leases),
            "paid_call_max_active": displayed_paid_call_max_active,
            "paid_call_rate_limit_cooldown_count": len(rate_limit_cooldowns),
            "paid_call_next_cooldown_seconds": min(
                [
                    cooldown.get("remaining_seconds")
                    for cooldown in rate_limit_cooldowns
                    if isinstance(cooldown.get("remaining_seconds"), (int, float))
                ],
                default=None,
            ),
        },
        "latest_events": deduped_latest_events,
        "evidence_feed": sorted(evidence_feed, key=_event_sort_key)[-MAX_LIVE_EVIDENCE_ITEMS:],
        "ledger_warnings": ledger_warnings,
        "operator": _suite_inventory(),
    }


def _contract_summary(contracts: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "count": len(contracts),
        "attention_count": sum(1 for contract in contracts if contract.get("attention")),
        "expected_units": sum(int(contract.get("expected_units") or 0) for contract in contracts),
        "complete_units": sum(int(contract.get("complete_units") or 0) for contract in contracts),
        "active_control_count": sum(
            1 for contract in contracts if (contract.get("control") or {}).get("active")
        ),
    }


def _dashboard_summary_data(
    data: dict[str, Any],
    *,
    scope: str = "latest",
) -> dict[str, Any]:
    """Strip detail records and scope the first-paint payload server-side."""
    summary = dict(data)
    raw_groups = [group for group in data.get("groups") or [] if isinstance(group, dict)]
    summary["run_index"] = [
        {
            key: group.get(key)
            for key in (
                "run_id",
                "updated_at",
                "latest_event_at",
                "elapsed",
                "severity",
                "module_count",
                "attention_count",
                "running_count",
                "ready_count",
            )
            if group.get(key) is not None
        }
        for group in raw_groups[:100]
    ]
    summary["run_index_total"] = len(raw_groups)
    resolved_scope = _resolved_run_scope(data, scope)
    member_run_ids = _scope_member_run_ids(data, resolved_scope)

    def matches_scope(item: dict[str, Any]) -> bool:
        if resolved_scope == "all":
            return True
        item_scope = str(
            item.get("run_id") or item.get("group") or item.get("path_group_id") or ""
        )
        if member_run_ids:
            return item_scope in member_run_ids
        return not resolved_scope or item_scope == resolved_scope

    compact_groups: list[dict[str, Any]] = []
    for group in raw_groups:
        if not matches_scope(group):
            continue
        compact_group = dict(group)
        group_contracts = [
            contract
            for contract in group.get("contracts") or []
            if isinstance(contract, dict)
        ]
        compact_group["contract_summary"] = _contract_summary(group_contracts)
        compact_group.pop("contracts", None)
        compact_group.pop("plans", None)
        compact_group.pop("schedulers", None)
        compact_modules: list[dict[str, Any]] = []
        for module in group.get("modules") or []:
            if not isinstance(module, dict):
                continue
            compact_module = dict(module)
            compact_module.pop("recent_events", None)
            compact_module.pop("event_model_progress", None)
            compact_modules.append(compact_module)
        compact_group["modules"] = compact_modules
        compact_groups.append(compact_group)
    summary["groups"] = compact_groups
    summary["contracts"] = []
    summary["plans"] = []
    summary["schedulers"] = []
    summary["evidence_feed"] = []
    summary["detail_loading"] = True
    summary["resolved_scope"] = resolved_scope
    summary["latest_events"] = [
        event
        for event in data.get("latest_events") or []
        if isinstance(event, dict) and matches_scope(event)
    ]

    flow = data.get("flow") if isinstance(data.get("flow"), dict) else {}
    compact_lanes: list[dict[str, Any]] = []
    for lane in flow.get("lanes") or []:
        if not isinstance(lane, dict):
            continue
        compact_lane = dict(lane)
        compact_items: list[dict[str, Any]] = []
        for item in lane.get("items") or []:
            if not isinstance(item, dict):
                continue
            if not matches_scope(item):
                continue
            compact_item = dict(item)
            for key in (
                "execute_command",
                "latest_model_response",
                "latest_transcript",
                "latest_user_message",
            ):
                compact_item.pop(key, None)
            compact_items.append(compact_item)
        compact_lane["items"] = compact_items
        compact_lanes.append(compact_lane)
    summary["flow"] = {**flow, "lanes": compact_lanes}

    queue = data.get("operational_queue") if isinstance(data.get("operational_queue"), dict) else {}
    compact_stages: list[dict[str, Any]] = []
    for stage in queue.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        items = [
            item
            for item in stage.get("items") or []
            if isinstance(item, dict) and matches_scope(item)
        ]
        compact_stage = dict(stage)
        compact_stage["items"] = items
        compact_stage["count"] = len(items)
        compact_stage["group_count"] = _flow_work_group_count(items)
        for target, field in (
            ("units", "units"),
            ("expected_units", "expected_units"),
            ("completed_units", "completed_units"),
            ("generation_expected_units", "generation_expected_units"),
            ("generated_units", "generated_units"),
            ("expected_score_units", "expected_score_units"),
            ("scored_units", "scored_units"),
            ("excluded_score_units", "excluded_score_units"),
            ("score_completed_units", "score_completed_units"),
            ("judge_calls_expected", "judge_calls_expected"),
            ("judge_calls_completed", "judge_calls_completed"),
            ("active_units", "active_units"),
            ("attention_units", "attention_units"),
        ):
            compact_stage[target] = sum(_flow_unit_count(item.get(field)) for item in items)
        compact_stage["score_completed_units"] = sum(
            _flow_unit_count(item.get("score_completed_units"))
            or (
                _flow_unit_count(item.get("scored_units"))
                + _flow_unit_count(item.get("excluded_score_units"))
            )
            for item in items
        )
        compact_stages.append(compact_stage)
    summary["operational_queue"] = {
        **queue,
        "stages": compact_stages,
        "total_units": sum(stage["generation_expected_units"] for stage in compact_stages),
        "generated_units": sum(stage["generated_units"] for stage in compact_stages),
        "active_units": sum(stage["active_units"] for stage in compact_stages),
        "attention_units": sum(stage["attention_units"] for stage in compact_stages),
        "generation_expected_units": sum(stage["generation_expected_units"] for stage in compact_stages),
        "generation_completed_units": sum(stage["generated_units"] for stage in compact_stages),
        "score_expected_units": sum(stage["expected_score_units"] for stage in compact_stages),
        "score_completed_units": sum(stage["score_completed_units"] for stage in compact_stages),
        "judge_calls_expected": sum(stage["judge_calls_expected"] for stage in compact_stages),
        "judge_calls_completed": sum(stage["judge_calls_completed"] for stage in compact_stages),
    }
    scoped_modules = [
        module
        for group in compact_groups
        for module in group.get("modules") or []
        if isinstance(module, dict)
    ]
    active_modules = [
        module
        for module in scoped_modules
        if not _is_rejected_from_analysis(module.get("disposition") or {})
    ]
    rejected_modules = [
        module
        for module in scoped_modules
        if _is_rejected_from_analysis(module.get("disposition") or {})
    ]
    scoped_contracts = [
        group.get("contract_summary") or {}
        for group in compact_groups
        if isinstance(group.get("contract_summary"), dict)
    ]
    latest_group = _select_latest_group(compact_groups)
    latest_event_at = next(
        (
            event.get("timestamp")
            for event in summary["latest_events"]
            if isinstance(event, dict) and event.get("timestamp")
        ),
        (latest_group or {}).get("latest_event_at"),
    )
    scoped_active_elapsed_seconds = max(
        [
            _flow_unit_count(module.get("elapsed_seconds"))
            for module in scoped_modules
            if module.get("severity") == "running"
        ],
        default=0,
    )
    scoped_suite_elapsed_seconds = max(
        [_flow_unit_count(group.get("elapsed_seconds")) for group in compact_groups],
        default=0,
    )
    scoped_summary = dict(data.get("summary") or {})
    scoped_summary.update(
        {
            "attention_count": sum(1 for module in active_modules if module.get("severity") == "attention"),
            "running_count": sum(1 for module in active_modules if module.get("severity") == "running"),
            "ready_count": sum(1 for module in active_modules if module.get("severity") == "ready"),
            "score_ready_count": sum(1 for module in active_modules if module.get("validity") == "score_ready"),
            "failed_count": sum(1 for module in active_modules if str(module.get("status") or "").startswith("failed")),
            "rejected_count": len(rejected_modules),
            "tracked_cost_total_usd": round(
                sum(
                    _flow_number((module.get("cost") or {}).get("total_cost_usd"))
                    for module in scoped_modules
                ),
                4,
            ),
            "latest_event_at": latest_event_at,
            "active_elapsed": (
                _format_duration(scoped_active_elapsed_seconds)
                if scoped_active_elapsed_seconds
                else "none"
            ),
            "latest_elapsed": (latest_group or {}).get("elapsed") or "none",
            "latest_run_id": (latest_group or {}).get("run_id"),
            "suite_elapsed": (
                _format_duration(scoped_suite_elapsed_seconds)
                if scoped_suite_elapsed_seconds
                else "none"
            ),
            "spend_attention_count": sum(
                1
                for module in scoped_modules
                if (module.get("spend_guard") or {}).get("severity") == "attention"
            ),
            "contract_count": sum(_flow_unit_count(item.get("count")) for item in scoped_contracts),
            "contract_attention_count": sum(
                _flow_unit_count(item.get("attention_count")) for item in scoped_contracts
            ),
            "contract_expected_units": sum(
                _flow_unit_count(item.get("expected_units")) for item in scoped_contracts
            ),
            "contract_complete_units": sum(
                _flow_unit_count(item.get("complete_units")) for item in scoped_contracts
            ),
            "scheduler_completed_units": summary["operational_queue"]["generation_completed_units"],
            "scheduler_active_units": summary["operational_queue"]["active_units"],
            "scheduler_attention_units": summary["operational_queue"]["attention_units"],
            "scheduler_running_count": sum(
                stage["count"]
                for stage in compact_stages
                if stage.get("id") in {"generating", "scoring"}
            ),
            "scheduler_queued_count": sum(
                stage["count"] for stage in compact_stages if stage.get("id") == "queued"
            ),
            "scheduler_attention_count": sum(
                stage["count"] for stage in compact_stages if stage.get("id") == "attention"
            ),
        }
    )
    summary["summary"] = scoped_summary
    summary["module_count"] = len(scoped_modules)
    return summary


def _resolved_run_scope(data: dict[str, Any], raw_scope: str | None) -> str:
    scope = str(raw_scope or "latest").strip()
    if scope == "all":
        return "all"
    if scope.startswith(("family:", "workflow:")):
        if scope == "workflow:active" and not _scope_member_run_ids(data, scope):
            return str((data.get("summary") or {}).get("latest_run_id") or "")
        return scope
    if scope == "latest":
        return str((data.get("summary") or {}).get("latest_run_id") or "")
    return scope


def _scope_member_run_ids(data: dict[str, Any], resolved_scope: str) -> set[str]:
    """Member run ids for a family or companion-workflow scope."""
    collections: list[dict[str, Any]] = []
    if resolved_scope.startswith("family:"):
        collections = [
            family
            for family in data.get("families") or []
            if isinstance(family, dict)
            and str(family.get("key") or "") == resolved_scope
        ]
    elif resolved_scope.startswith("workflow:"):
        collections = [
            workflow
            for workflow in data.get("workflows") or []
            if isinstance(workflow, dict)
            and str(workflow.get("key") or "") == resolved_scope
        ]
    for collection in collections:
        return {str(run_id) for run_id in collection.get("member_run_ids") or []}
    return set()


def _canonical_detail_stage(value: Any) -> str:
    stage = str(value or "event").lower().replace("-", "_").replace(" ", "_")
    return {
        "generation": "generating",
        "run": "generating",
        "scoring": "judging",
        "ready": "score_ready",
        "failed": "attention",
        "rejected_from_analysis": "rejected",
    }.get(stage, stage)


def _dashboard_evidence_payload(
    data: dict[str, Any],
    *,
    scope: str = "latest",
    stage: str = "all",
    content: str = "all",
    window: str = "100",
    module: str = "",
) -> dict[str, Any]:
    resolved_scope = _resolved_run_scope(data, scope)
    requested_stage = _canonical_detail_stage(stage) if stage != "all" else "all"
    requested_content = content if content in {"all", "text", "writes"} else "all"
    requested_module = str(module or "").strip()
    items = [item for item in data.get("evidence_feed") or [] if isinstance(item, dict)]
    member_run_ids = _scope_member_run_ids(data, resolved_scope)
    if resolved_scope != "all":
        items = [
            item
            for item in items
            if (
                str(item.get("group") or item.get("run_id") or "") in member_run_ids
                if member_run_ids
                else str(item.get("group") or item.get("run_id") or "")
                == resolved_scope
            )
        ]
    if requested_module:
        items = [
            item
            for item in items
            if str(item.get("module_path") or item.get("module") or "") == requested_module
        ]
    if requested_stage != "all":
        items = [
            item
            for item in items
            if _canonical_detail_stage("attention" if item.get("problem") else item.get("stage"))
            == requested_stage
        ]
    if requested_content == "writes":
        items = [item for item in items if item.get("kind") != "turn_pair"]
    elif requested_content == "text":
        items = [
            item
            for item in items
            if item.get("kind") == "turn_pair"
            or _canonical_detail_stage("attention" if item.get("problem") else item.get("stage"))
            == "attention"
        ]
    total_count = len(items)
    if window != "all":
        try:
            window_size = int(window)
        except (TypeError, ValueError):
            window_size = 100
        window_size = max(1, min(MAX_DETAIL_WINDOW, window_size))
        items = items[-window_size:]
    return {
        "schema_version": "benchmark-live-dashboard-evidence-v1",
        "generated_at": data.get("generated_at"),
        "scope": scope,
        "resolved_scope": resolved_scope,
        "stage": requested_stage,
        "content": requested_content,
        "window": window,
        "module": requested_module,
        "total_count": total_count,
        "items": items,
    }


def _dashboard_contract_payload(
    data: dict[str, Any],
    *,
    scope: str = "latest",
) -> dict[str, Any]:
    resolved_scope = _resolved_run_scope(data, scope)
    contracts = [
        contract
        for contract in data.get("contracts") or []
        if isinstance(contract, dict)
    ]
    member_run_ids = _scope_member_run_ids(data, resolved_scope)
    if resolved_scope != "all":
        contracts = [
            contract
            for contract in contracts
            if (
                str(contract.get("path_group_id") or contract.get("run_id") or "")
                in member_run_ids
                if member_run_ids
                else str(contract.get("path_group_id") or contract.get("run_id") or "")
                == resolved_scope
            )
        ]
    return {
        "schema_version": "benchmark-live-dashboard-contracts-v1",
        "generated_at": data.get("generated_at"),
        "scope": scope,
        "resolved_scope": resolved_scope,
        "summary": _contract_summary(contracts),
        "contracts": contracts,
    }


def _detail_scope_root(
    summary_data: dict[str, Any],
    *,
    results_root: Path,
    scope: str,
) -> tuple[str, Path | None]:
    resolved_scope = _resolved_run_scope(summary_data, scope)
    root = results_root.resolve()
    if resolved_scope == "all":
        return resolved_scope, root
    if _scope_member_run_ids(summary_data, resolved_scope):
        return resolved_scope, root
    if not resolved_scope or Path(resolved_scope).name != resolved_scope:
        return resolved_scope, None
    candidate = (root / resolved_scope).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return resolved_scope, None
    return resolved_scope, candidate if candidate.is_dir() else None


def _build_dashboard_evidence_detail(
    summary_data: dict[str, Any],
    *,
    results_root: Path,
    scope: str,
    stage: str,
    content: str,
    window: str,
    module: str,
) -> dict[str, Any]:
    """Build evidence from only the selected run unless all runs are requested."""
    resolved_scope = _resolved_run_scope(summary_data, scope)
    member_run_ids = _scope_member_run_ids(summary_data, resolved_scope)
    if member_run_ids:
        detail_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "evidence_feed": [],
        }
        root = results_root.resolve()
        for run_id in sorted(member_run_ids):
            member_root = (root / run_id).resolve()
            try:
                member_root.relative_to(root)
            except ValueError:
                continue
            if not member_root.is_dir():
                continue
            member_data = build_dashboard_data(
                DashboardOptions(results_root=member_root)
            )
            for raw_item in member_data.get("evidence_feed") or []:
                if not isinstance(raw_item, dict):
                    continue
                detail_data["evidence_feed"].append(
                    {**raw_item, "group": run_id, "run_id": run_id}
                )
        detail_data["evidence_feed"].sort(key=_event_sort_key)
    else:
        resolved_scope, detail_root = _detail_scope_root(
            summary_data,
            results_root=results_root,
            scope=scope,
        )
        if detail_root is None:
            detail_data = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "evidence_feed": [],
            }
        else:
            detail_data = build_dashboard_data(
                DashboardOptions(results_root=detail_root)
            )
        if detail_root is not None and resolved_scope != "all":
            normalized_items = []
            for raw_item in detail_data.get("evidence_feed") or []:
                if not isinstance(raw_item, dict):
                    continue
                item = dict(raw_item)
                item["group"] = resolved_scope
                item["run_id"] = resolved_scope
                normalized_items.append(item)
            detail_data["evidence_feed"] = normalized_items
    detail_data["families"] = summary_data.get("families") or []
    detail_data["workflows"] = summary_data.get("workflows") or []
    detail_scope = scope if _scope_member_run_ids(summary_data, resolved_scope) else "all"
    payload = _dashboard_evidence_payload(
        detail_data,
        scope=detail_scope,
        stage=stage,
        content=content,
        window=window,
        module=module,
    )
    payload["scope"] = scope
    payload["resolved_scope"] = resolved_scope
    return payload


def _build_dashboard_contract_detail(
    summary_data: dict[str, Any],
    *,
    results_root: Path,
    scope: str,
) -> dict[str, Any]:
    """Resolve contract artifacts only inside the selected run."""
    resolved_scope = _resolved_run_scope(summary_data, scope)
    member_run_ids = _scope_member_run_ids(summary_data, resolved_scope)
    if member_run_ids:
        contracts = []
        root = results_root.resolve()
        for run_id in sorted(member_run_ids):
            member_root = (root / run_id).resolve()
            try:
                member_root.relative_to(root)
            except ValueError:
                continue
            if not member_root.is_dir():
                continue
            contracts.extend(
                {**contract, "path_group_id": run_id}
                for contract in _load_contract_summaries(member_root)
                if isinstance(contract, dict)
            )
    else:
        resolved_scope, detail_root = _detail_scope_root(
            summary_data,
            results_root=results_root,
            scope=scope,
        )
        contracts = (
            _load_contract_summaries(detail_root)
            if detail_root is not None
            else []
        )
    if not member_run_ids and resolved_scope != "all":
        contracts = [
            {**contract, "path_group_id": resolved_scope}
            for contract in contracts
            if isinstance(contract, dict)
        ]
    detail_scope = scope if member_run_ids else "all"
    payload = _dashboard_contract_payload(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "contracts": contracts,
            "families": summary_data.get("families") or [],
            "workflows": summary_data.get("workflows") or [],
        },
        scope=detail_scope,
    )
    payload["scope"] = scope
    payload["resolved_scope"] = resolved_scope
    return payload


def render_html(
    *,
    title: str = "Anti-sycophancy Benchmark Suite",
    poll_ms: int = 2500,
    csrf_token: str = "",
) -> str:
    safe_title = html.escape(title)
    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <script src="/assets/theme-init.js"></script>
  <link rel="stylesheet" href="/assets/dashboard.css">
</head>
<body data-poll-ms="__POLL_MS__" data-csrf-token="__CSRF_TOKEN__">
  <nav class="top-menu" aria-label="Benchmark quick actions">
    <div class="top-brand">
      __BRAND_LOGO__
      <span>Anti-sycophancy</span>
    </div>
    <div class="top-scope" id="topScopeControl" aria-label="Benchmark run scope">
      <span class="top-scope-label">Scope</span>
      <span class="top-scope-placeholder">Current workflow</span>
    </div>
    <div class="top-status" aria-label="Run summary while scrolling">
      <div class="top-stat" id="topCompleteStat"><span data-short="Stage">Stage</span><strong id="topComplete">--</strong></div>
      <div class="top-stat" id="topElapsedStat"><span id="topElapsedLabel" data-short="Time">Run time</span><strong id="topElapsed">--</strong></div>
      <div class="top-stat running" id="topActiveStat"><span data-short="Live">Active</span><strong id="topActive">--</strong></div>
      <button class="top-stat top-stat-button attention" id="topErrorsStat" type="button" data-quick-stage="attention" aria-pressed="false" title="Filter the evidence feed to failed, stale, malformed, or incomplete items."><span data-short="Issues">Issues</span><strong id="topErrors">--</strong></button>
    </div>
    <div class="quick-actions" aria-label="Dashboard quick actions">
      <button class="quick-action utility" type="button" data-quick-action="follow-live" id="followLiveToggle" aria-pressed="true" title="Pause or resume evidence feed auto-follow.">Pause feed</button>
    </div>
    <button class="theme-toggle" type="button" id="themeToggle" aria-label="Toggle color theme" title="Toggle color theme">
      <span aria-hidden="true" id="themeGlyph">◐</span>
      <span class="theme-toggle-label" id="themeLabel">Theme follows system</span>
    </button>
  </nav>
  <header class="masthead">
    <div>
      <p class="eyebrow">Live Benchmark Dashboard</p>
      <h1>__TITLE__</h1>
      <div class="masthead-copy">Read-only run ledger over RUN_CONTRACT.json, RUN_STATUS.json, and RUN_EVENTS.jsonl.</div>
    </div>
    <div class="live-meta">
      <div class="meta-actions">
        <div class="live-badge"><span class="live-dot"></span><span id="liveLabel">Watching ledgers</span></div>
      </div>
      <div class="last-refresh" id="lastRefresh">Waiting for first refresh</div>
    </div>
  </header>
  <main id="app"><div class="empty">Loading run ledgers...</div></main>
  <aside class="copy-panel" id="copyPanel" hidden>
    <div class="copy-panel-head">
      <div>
        <h2 id="copyPanelTitle">Copy text</h2>
        <div class="panel-kicker" id="copyPanelNote">The selected text is shown here for browser clipboard fallbacks.</div>
      </div>
      <button class="copy-button" type="button" id="closeCopyPanel">Close</button>
    </div>
    <textarea class="copy-textarea" id="copyTextarea" readonly></textarea>
    <div class="action-row">
      <button class="copy-button" type="button" id="selectCopyText">Select text</button>
    </div>
  </aside>
  <script src="/assets/dashboard.js" defer></script>
</body>
</html>
"""
    static_logo_src = html.escape(_asset_url(f"/assets/{BRAND_LOGO_STATIC_NAME}"))
    running_logo_src = html.escape(_asset_url(f"/assets/{BRAND_LOGO_RUNNING_NAME}"))
    logo_markup = (
        (
            f'<img id="brandShield" class="brand-shield" src="{static_logo_src}" '
            f'data-static-src="{static_logo_src}" data-running-src="{running_logo_src}" '
            'alt="Anti-sycophancy shield">'
        )
        if static_logo_src
        else '<span class="brand-shield" aria-hidden="true"></span>'
    )
    return (
        page.replace("__TITLE__", safe_title)
        .replace("__POLL_MS__", str(int(poll_ms)))
        .replace("__CSRF_TOKEN__", html.escape(csrf_token, quote=True))
        .replace("__BRAND_LOGO__", logo_markup)
    )


class DashboardHandler(BaseHTTPRequestHandler):
    options: DashboardOptions
    page_title: str
    poll_ms: int
    csrf_token: str = ""
    operator_id: str = "local:unknown"
    store: DashboardStore | None = None
    _runs_cache: dict[str, tuple[float, str, dict[str, Any], bytes]] = {}
    _runs_cache_lock = threading.Lock()
    _runs_builds_in_progress: dict[str, threading.Event] = {}

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[live-dashboard] {self.address_string()} - {fmt % args}", file=sys.stderr)

    def _host_is_local(self) -> bool:
        """Reject non-local Host headers (browser DNS-rebinding/CSRF guard)."""
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("[") and "]" in host:
            hostname = host[1 : host.index("]")]
        elif host.startswith("["):
            return False
        elif host.count(":") == 1:
            hostname = host.split(":", 1)[0]
        else:
            hostname = host
        if hostname == "localhost":
            return True
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False

    def _peer_is_local(self) -> bool:
        try:
            return ipaddress.ip_address(self.client_address[0]).is_loopback
        except (ValueError, IndexError):
            return False

    def _request_is_local(self) -> bool:
        return self._host_is_local() and self._peer_is_local()

    def _reject_nonlocal_request(self, *, include_body: bool = True) -> None:
        self._send(
            b'{"ok":false,"error":"non-local dashboard request rejected"}',
            content_type="application/json; charset=utf-8",
            status=403,
            include_body=include_body,
        )

    def _send(
        self,
        body: bytes,
        *,
        content_type: str,
        status: int = 200,
        include_body: bool = True,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        if content_type:
            self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; font-src 'self'; "
            "img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
        )
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload, indent=2, default=str).encode()
        self._send(body, content_type="application/json; charset=utf-8", status=status)

    def _build_runs_data(self, *, summary_only: bool = False) -> dict[str, Any]:
        try:
            return build_dashboard_data(
                DashboardOptions(
                    results_root=self.options.results_root,
                    event_limit=self.options.event_limit,
                    summary_only=summary_only,
                )
            )
        except Exception as exc:
            now = datetime.now(timezone.utc).isoformat()
            return {
                "schema_version": "benchmark-live-dashboard-v1",
                "generated_at": now,
                "results_root": _relative(self.options.results_root, REPO_ROOT),
                "groups": [],
                "module_count": 0,
                "contracts": [],
                "plans": [],
                "schedulers": [],
                "flow": {"lanes": [], "counts": {}},
                "summary": {},
                "latest_events": [],
                "operator": {},
                "error": sanitize_error_message(exc),
            }

    def _build_runs_payload(self) -> bytes:
        data = _dashboard_summary_data(self._build_runs_data(summary_only=True))
        return json.dumps(data, separators=(",", ":"), default=str).encode()

    def _cached_dashboard_entry(
        self,
        revision: str | None = None,
        *,
        summary_only: bool = False,
    ) -> tuple[dict[str, Any], bytes]:
        base_cache_key = _runs_cache_key(self.options.results_root)
        cache_key = f"{base_cache_key}:summary" if summary_only else base_cache_key
        source_revision = revision or _dashboard_source_revision(self.options.results_root)
        with DashboardHandler._runs_cache_lock:
            cached = DashboardHandler._runs_cache.get(cache_key)
            if cached and cached[1] == source_revision:
                return cached[2], cached[3]
            build_event = DashboardHandler._runs_builds_in_progress.get(cache_key)
            if build_event is None:
                build_event = threading.Event()
                DashboardHandler._runs_builds_in_progress[cache_key] = build_event
                builder = True
            else:
                if cached and cached[1] == source_revision:
                    return cached[2], cached[3]
                builder = False

        if not builder:
            build_event.wait(timeout=BUILD_WAIT_TIMEOUT_SECONDS)
            with DashboardHandler._runs_cache_lock:
                cached = DashboardHandler._runs_cache.get(cache_key)
                if cached and cached[1] == source_revision:
                    return cached[2], cached[3]
            build_event = threading.Event()

        data: dict[str, Any] | None = None
        body: bytes | None = None
        try:
            data = self._build_runs_data(summary_only=summary_only)
            body = json.dumps(
                _dashboard_summary_data(data),
                separators=(",", ":"),
                default=str,
            ).encode()
        finally:
            with DashboardHandler._runs_cache_lock:
                if data is not None and body is not None and not data.get("error"):
                    DashboardHandler._runs_cache[cache_key] = (
                        time.monotonic(),
                        source_revision,
                        data,
                        body,
                    )
                if DashboardHandler._runs_builds_in_progress.get(cache_key) is build_event:
                    DashboardHandler._runs_builds_in_progress.pop(cache_key, None)
            build_event.set()
        return data, body

    def _cached_runs_payload(self, revision: str | None = None) -> bytes:
        _data, body = self._cached_dashboard_entry(revision, summary_only=True)
        return body

    def _cached_dashboard_data(self, revision: str | None = None) -> dict[str, Any]:
        data, _body = self._cached_dashboard_entry(revision)
        return data

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0:
            return {}
        if length > 65536:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _resolve_status_path(self, raw_path: Any) -> Path:
        if not raw_path:
            raise ValueError("status_path is required")
        status_path = Path(str(raw_path))
        candidates = [status_path] if status_path.is_absolute() else [REPO_ROOT / status_path]
        if not status_path.is_absolute():
            parts = status_path.parts
            if "results" in parts:
                suffix = list(parts[parts.index("results") + 1 :])
                if suffix and suffix[0] == self.options.results_root.name:
                    suffix = suffix[1:]
                if suffix:
                    candidates.append(self.options.results_root.joinpath(*suffix))
            candidates.append(self.options.results_root / status_path)
        candidate = next((item for item in candidates if item.exists()), candidates[0])
        resolved = candidate.resolve()
        results_root = self.options.results_root.resolve()
        try:
            resolved.relative_to(results_root)
        except ValueError as exc:
            raise ValueError("status_path must be inside the watched results root") from exc
        if resolved.name != "RUN_STATUS.json":
            raise ValueError("status_path must point to RUN_STATUS.json")
        if not resolved.exists():
            raise ValueError("RUN_STATUS.json does not exist")
        return resolved

    def _write_disposition(self) -> None:
        try:
            request = self._read_json_body()
            status_path = self._resolve_status_path(request.get("status_path"))
            disposition_value = str(request.get("disposition") or "").strip()
            if disposition_value not in {"rejected_from_analysis", "candidate"}:
                raise ValueError("disposition must be rejected_from_analysis or candidate")
            status_payload = sanitize_ledger_value(_load_json(status_path))
            if (
                disposition_value == "rejected_from_analysis"
                and not _status_allows_analysis_rejection(status_payload)
            ):
                raise ValueError(
                    "only failed, nonpublishable runs can be rejected from analysis"
                )
            reason = sanitize_error_message(str(request.get("reason") or "operator_disposition_update"))
            notes = sanitize_error_message(str(request.get("notes") or ""))
        except ValueError as exc:
            self._send_json({"ok": False, "error": sanitize_error_message(exc)}, status=400)
            return

        if disposition_value == "rejected_from_analysis":
            disposition = {
                "schema_version": DISPOSITION_SCHEMA_VERSION,
                "disposition": "rejected_from_analysis",
                "reason": reason,
                "eligible_for_generation": False,
                "eligible_for_scoring": False,
                "eligible_for_promotion": False,
                "decided_at": utc_now(),
                "decided_by": self.operator_id,
                "notes": notes,
                "source_status": {
                    "status": status_payload.get("status"),
                    "validity": status_payload.get("validity"),
                    "stage": status_payload.get("stage"),
                    "failure_reason": sanitize_error_message(str(status_payload.get("failure_reason") or "")),
                },
            }
        else:
            eligibility = _disposition_eligibility(status_payload, {})
            disposition = {
                "schema_version": DISPOSITION_SCHEMA_VERSION,
                "disposition": "candidate",
                "reason": reason,
                **eligibility,
                "decided_at": utc_now(),
                "decided_by": self.operator_id,
                "notes": notes,
            }

        disposition_path = status_path.parent / DISPOSITION_FILENAME
        action = "reject" if disposition_value == "rejected_from_analysis" else "restore"
        _persist_disposition_update(
            status_path.parent / DISPOSITION_EVENTS_FILENAME,
            {
                "schema_version": "benchmark-run-disposition-event-v1",
                "event": "run_disposition",
                "action": action,
                "disposition": disposition_value,
                "operator_id": self.operator_id,
                "decided_at": disposition["decided_at"],
                "reason": reason,
                "status_path": _relative(status_path, REPO_ROOT),
            },
            disposition_path,
            disposition,
        )
        with DashboardHandler._runs_cache_lock:
            cache_key = str(self.options.results_root.resolve())
            DashboardHandler._runs_cache.pop(cache_key, None)
            DashboardHandler._runs_cache.pop(f"{cache_key}:summary", None)
        if self.store is not None:
            self.store.refresh_async(force=True)
        self._send_json(
            {
                "ok": True,
                "disposition": disposition_value,
                "path": _relative(disposition_path, REPO_ROOT),
            }
        )

    def _response_for_path(self) -> tuple[bytes, str, int]:
        parsed = urlparse(self.path)
        asset = DASHBOARD_ASSETS.get(parsed.path)
        if asset:
            asset_path, content_type = asset
            try:
                return asset_path.read_bytes(), content_type, 200
            except OSError:
                return b"not found\n", "text/plain; charset=utf-8", 404
        if parsed.path == "/":
            body = render_html(
                title=self.page_title,
                poll_ms=self.poll_ms,
                csrf_token=self.csrf_token,
            ).encode()
            return body, "text/html; charset=utf-8", 200
        if parsed.path == "/api/runs":
            if self.store is not None:
                body = json.dumps(
                    _dashboard_summary_data(self.store.summary()),
                    separators=(",", ":"),
                    default=str,
                ).encode()
                return body, "application/json; charset=utf-8", 200
            return self._cached_runs_payload(), "application/json; charset=utf-8", 200
        return b"not found\n", "text/plain; charset=utf-8", 404

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if not self._request_is_local():
            self._reject_nonlocal_request()
            return
        if parsed.path in {"/api/runs", "/api/evidence", "/api/contracts"}:
            if self.store is not None:
                data = self.store.summary()
                revision = str(data.get("source_revision") or "initializing")
            else:
                data = None
                revision = _dashboard_source_revision(self.options.results_root)
            query = parse_qs(parsed.query)
            scope = (query.get("scope") or ["latest"])[0]
            detail_key = ""
            if parsed.path != "/api/runs" or query:
                normalized_query = "&".join(
                    f"{key}={','.join(sorted(values))}"
                    for key, values in sorted(query.items())
                )
                detail_key = hashlib.blake2b(
                    f"{parsed.path}?{normalized_query}".encode(),
                    digest_size=8,
                ).hexdigest()
            refresh_health = ""
            if data is not None and (data.get("refresh_error") or data.get("refreshing")):
                refresh_health = hashlib.blake2b(
                    f"{data.get('refresh_error') or ''}:{bool(data.get('refreshing'))}".encode(),
                    digest_size=4,
                ).hexdigest()
            etag_suffix = "-".join(part for part in (detail_key, refresh_health) if part)
            etag = f'"{revision}{f"-{etag_suffix}" if etag_suffix else ""}"'
            if self.headers.get("If-None-Match") == etag:
                self._send(
                    b"",
                    content_type="",
                    status=304,
                    headers={"ETag": etag},
                )
                return
            if parsed.path == "/api/runs":
                if data is None and scope == "latest":
                    body = self._cached_runs_payload(revision)
                else:
                    if data is None:
                        data, _summary_body = self._cached_dashboard_entry(
                            revision,
                            summary_only=True,
                        )
                    body = json.dumps(
                        _dashboard_summary_data(data, scope=scope),
                        separators=(",", ":"),
                        default=str,
                    ).encode()
            else:
                if data is None:
                    data, _summary_body = self._cached_dashboard_entry(
                        revision,
                        summary_only=True,
                    )
                if parsed.path == "/api/evidence":
                    payload = _build_dashboard_evidence_detail(
                        data,
                        results_root=self.options.results_root,
                        scope=scope,
                        stage=(query.get("stage") or ["all"])[0],
                        content=(query.get("content") or ["all"])[0],
                        window=(query.get("window") or ["100"])[0],
                        module=(query.get("module") or [""])[0],
                    )
                else:
                    payload = _build_dashboard_contract_detail(
                        data,
                        results_root=self.options.results_root,
                        scope=scope,
                    )
                body = json.dumps(payload, separators=(",", ":"), default=str).encode()
            self._send(
                body,
                content_type="application/json; charset=utf-8",
                headers={"ETag": etag},
            )
            return
        body, content_type, status = self._response_for_path()
        self._send(body, content_type=content_type, status=status)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not self._request_is_local():
            self._reject_nonlocal_request()
            return
        if parsed.path == "/api/disposition":
            csrf_token = self.headers.get("X-Benchmark-CSRF") or ""
            if not self.csrf_token or not secrets.compare_digest(csrf_token, self.csrf_token):
                self._send_json({"ok": False, "error": "CSRF token rejected"}, status=403)
                return
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if content_type != "application/json":
                self._send_json(
                    {"ok": False, "error": "Content-Type must be application/json"},
                    status=415,
                )
                return
            self._write_disposition()
            return
        self._send(b"not found\n", content_type="text/plain; charset=utf-8", status=404)

    def do_HEAD(self) -> None:
        if not self._request_is_local():
            self._reject_nonlocal_request(include_body=False)
            return
        body, content_type, status = self._response_for_path()
        self._send(body, content_type=content_type, status=status, include_body=False)


def _dashboard_results_root_error(results_root: Path) -> str | None:
    """Reject a lifecycle parent that would collapse buckets into pseudo-runs."""
    root = Path(results_root)
    if root.name != "results" or not root.is_dir():
        return None
    populated = []
    for name in RESULTS_LIFECYCLE_DIRS:
        lifecycle_root = root / name
        if not lifecycle_root.is_dir():
            continue
        if any(lifecycle_root.rglob("RUN_STATUS.json")) or any(
            lifecycle_root.rglob(CONTRACT_FILENAME)
        ):
            populated.append(name)
    if not populated:
        return None
    buckets = ", ".join(populated)
    return (
        f"{root} contains lifecycle buckets ({buckets}). Watching their parent "
        "would group those bucket names as pseudo-runs. Choose one lifecycle root, "
        "for example --results-root results/prepared for publication runs or "
        "--results-root results/testing for smoke runs."
    )


def run_server(
    *,
    host: str,
    port: int,
    options: DashboardOptions,
    title: str,
    poll_ms: int,
    operator_id: str | None = None,
) -> None:
    _require_loopback_address(host, field="host")
    resolved_operator_id = (operator_id or _default_operator_id()).strip()
    if not resolved_operator_id:
        raise ValueError("operator_id must not be empty")
    root_error = _dashboard_results_root_error(options.results_root)
    if root_error:
        raise SystemExit(root_error)
    store = DashboardStore(
        results_root=options.results_root,
        build_summary=lambda: build_dashboard_data(
            DashboardOptions(
                results_root=options.results_root,
                event_limit=options.event_limit,
                summary_only=True,
            )
        ),
        source_revision=lambda: _dashboard_source_revision(options.results_root),
    )
    handler = type(
        "ConfiguredDashboardHandler",
        (DashboardHandler,),
        {
            "options": options,
            "page_title": title,
            "poll_ms": poll_ms,
            "csrf_token": secrets.token_urlsafe(32),
            "operator_id": resolved_operator_id,
            "store": store,
        },
    )
    server_class = ThreadingHTTPServer
    if ipaddress.ip_address(host).version == 6:
        class LoopbackIPv6DashboardServer(ThreadingHTTPServer):
            address_family = socket.AF_INET6

        server_class = LoopbackIPv6DashboardServer
    try:
        server = server_class((host, port), handler)
    except OSError as exc:
        raise SystemExit(
            f"Could not start live dashboard on {host}:{port}: {exc}. "
            "Choose another port with --port."
        ) from exc
    store.start(refresh_interval_seconds=max(0.25, poll_ms / 1000))
    url_host = f"[{host}]" if ":" in host else host
    url = f"http://{url_host}:{port}"
    print(f"Serving benchmark live dashboard at {url}")
    print(f"Watching: {options.results_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping benchmark live dashboard.")
    finally:
        server.server_close()
        store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a local operator cockpit for benchmark run ledgers.")
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help=(
            "One lifecycle directory containing benchmark runs "
            "(for example results/prepared or results/testing)."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local dashboard server.")
    parser.add_argument(
        "--operator-id",
        default=_default_operator_id(),
        help="Attribution recorded for dashboard disposition events (default: local OS user).",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port for the local dashboard server.")
    parser.add_argument("--event-limit", type=int, default=80, help="Number of recent events to read per module.")
    parser.add_argument("--poll-ms", type=int, default=2500, help="Browser refresh interval in milliseconds.")
    parser.add_argument("--title", default="Anti-sycophancy Benchmark Suite", help="Dashboard page title.")
    parser.add_argument("--once", action="store_true", help="Print dashboard JSON and exit instead of serving HTML.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = DashboardOptions(results_root=Path(args.results_root), event_limit=args.event_limit)
    root_error = _dashboard_results_root_error(options.results_root)
    if root_error:
        raise SystemExit(root_error)
    if args.once:
        print(json.dumps(build_dashboard_data(options), indent=2, default=str))
        return 0
    run_server(
        host=args.host,
        port=args.port,
        options=options,
        title=args.title,
        poll_ms=args.poll_ms,
        operator_id=args.operator_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
