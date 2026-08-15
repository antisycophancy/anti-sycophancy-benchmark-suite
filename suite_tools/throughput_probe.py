"""Paid OpenRouter throughput probe for benchmark planning.

This is diagnostic tooling, not benchmark evidence. It sends tiny chat
completion calls, ramps concurrency per model, stops each model on the first
429, and writes a machine-readable report that can inform safe scheduler caps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import httpx

from suite_tools.env import load_repo_env_files
from suite_tools.credential_policy import require_credential_destination
from suite_tools.fake_provider import FakeOpenAIProvider
from suite_tools.openrouter_preflight import fetch_key_info, sanitize_key_info
from suite_tools.paid_call_lease import (
    is_rate_limit_error,
    paid_call_capacity_report,
    paid_call_lease,
    provider_from_base_url,
    rate_limit_delay_seconds,
    record_rate_limit_cooldown,
    set_paid_call_policy,
)
from suite_tools.run_monitor import atomic_write_json, sanitize_error_message, sanitize_ledger_value

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "throughput-probes"
DEFAULT_MODELS = (
    "google/gemini-3-flash-preview",
    "anthropic/claude-3-haiku",
)
DEFAULT_MAX_TOKENS = 16


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_steps(value: str) -> list[int]:
    """Parse concurrency steps like ``1,2,3,4``."""
    steps: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed <= 0:
            raise ValueError("concurrency steps must be positive integers")
        if parsed not in steps:
            steps.append(parsed)
    if not steps:
        raise ValueError("at least one concurrency step is required")
    return steps


def parse_models(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    if not models:
        raise ValueError("at least one model is required")
    return models


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return round(ordered[index], 4)


def _steady_state_utilization(
    intervals: list[tuple[float, float]],
    capacity: int,
) -> float:
    """Measure lease occupancy after warm-up and before final drain."""
    if not intervals or capacity <= 0:
        return 0.0
    grouped: dict[float, int] = {}
    for started, finished in intervals:
        if finished <= started:
            continue
        grouped[started] = grouped.get(started, 0) + 1
        grouped[finished] = grouped.get(finished, 0) - 1
    if not grouped:
        return 0.0

    threshold = max(1, math.ceil(capacity * 0.9))
    active = 0
    first_steady: float | None = None
    last_steady_end: float | None = None
    segments: list[tuple[float, float, int]] = []
    previous_time = min(grouped)
    for timestamp in sorted(grouped):
        if timestamp > previous_time:
            segments.append((previous_time, timestamp, active))
        previous_active = active
        active += grouped[timestamp]
        if previous_active < threshold <= active and first_steady is None:
            first_steady = timestamp
        if previous_active >= threshold > active:
            last_steady_end = timestamp
        previous_time = timestamp

    if first_steady is None or last_steady_end is None or last_steady_end <= first_steady:
        return 0.0
    occupied_seconds = sum(
        max(0.0, min(end, last_steady_end) - max(start, first_steady)) * active_count
        for start, end, active_count in segments
        if end > first_steady and start < last_steady_end
    )
    return round(
        min(1.0, occupied_seconds / ((last_steady_end - first_steady) * capacity)),
        4,
    )


def _usage_cost(response_json: dict[str, Any]) -> float:
    usage = response_json.get("usage") if isinstance(response_json.get("usage"), dict) else {}
    value = usage.get("cost")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rate_limit_kind(status_code: int | None, text: str, headers: dict[str, Any]) -> str | None:
    lowered = text.lower()
    if (
        status_code == 402
        or "insufficient_quota" in lowered
        or "insufficient quota" in lowered
        or "insufficient credits" in lowered
        or "billing_error" in lowered
    ):
        return None
    if status_code == 429 or "too-many-requests" in lowered or "rate limit" in lowered:
        if "token" in lowered:
            return "token_or_throughput_rate_limit"
        if any(key.lower() in {"x-ratelimit-remaining", "x-ratelimit-reset", "retry-after"} for key in headers):
            return "request_rate_limit"
        return "rate_limit"
    return None


def _failure_kind(status_code: int | None, text: str, headers: dict[str, Any]) -> str | None:
    """Classify non-success probe failures without treating all 4xx as limits."""
    lowered = text.lower()
    if (
        status_code == 402
        or "insufficient_quota" in lowered
        or "insufficient quota" in lowered
        or "insufficient credits" in lowered
        or "billing_error" in lowered
    ):
        return "billing"
    rate_limit_kind = _rate_limit_kind(status_code, text, headers)
    if rate_limit_kind is not None:
        return rate_limit_kind
    if status_code == 400 and "max_output_tokens" in lowered and "minimum" in lowered:
        return "token_parameter_minimum"
    if status_code in {401, 403}:
        return "auth_or_provider_access"
    if status_code in {500, 502, 503, 504}:
        return "provider_transient_failure"
    if status_code is not None:
        return f"http_{status_code}"
    return None


def _exception_failure_kind(exc: BaseException) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.NetworkError):
        return "connection_error"
    return "exception"


def _chat_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def run_one_call(
    *,
    model: str,
    url: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
    lease_dir: Path,
    step_concurrency: int,
    request_index: int,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    lease_acquired_at: float | None = None
    lease_finished_at: float | None = None

    def timing_fields() -> dict[str, float | None]:
        now = time.monotonic()
        acquired = lease_acquired_at or now
        finished = lease_finished_at or now
        return {
            "queue_wait_seconds": round(max(0.0, acquired - started), 4),
            "lease_hold_seconds": round(
                max(0.0, finished - acquired) if lease_acquired_at is not None else 0.0,
                4,
            ),
            "_lease_started_monotonic": acquired if lease_acquired_at is not None else None,
            "_lease_finished_monotonic": finished if lease_acquired_at is not None else None,
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "SUS Unified Benchmark Throughput Probe",
    }
    provider = provider_from_base_url(url)
    response_text = ""
    response_headers: dict[str, Any] = {}
    status_code: int | None = None
    cost = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    try:
        with paid_call_lease(
            provider=provider,
            model=model,
            role="throughput_probe",
            module="throughput_probe",
            run_id=lease_dir.parent.name,
            unit_id=f"{model}:c{step_concurrency}:r{request_index}",
            lease_dir=lease_dir,
            max_active_calls=step_concurrency,
            timeout_seconds=timeout_seconds + 30,
            poll_seconds=0.05,
        ):
            lease_acquired_at = time.monotonic()
            try:
                post = http_client.post if http_client is not None else httpx.post
                response = post(
                    url,
                    headers=headers,
                    json=_chat_payload(model, prompt, max_tokens),
                    timeout=timeout_seconds,
                )
            finally:
                lease_finished_at = time.monotonic()
        status_code = response.status_code
        response_text = response.text[:1000]
        response_headers = dict(response.headers)
        malformed_response = False
        try:
            response_json = response.json()
        except (json.JSONDecodeError, ValueError):
            response_json = {}
            malformed_response = status_code == 200
        if not isinstance(response_json, dict):
            response_json = {}
            malformed_response = status_code == 200
        usage = response_json.get("usage") if isinstance(response_json.get("usage"), dict) else {}
        try:
            prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        except (TypeError, ValueError):
            prompt_tokens = 0
            completion_tokens = 0
        if status_code == 200:
            choices = response_json.get("choices") if isinstance(response_json, dict) else None
            first_choice = choices[0] if isinstance(choices, list) and choices else None
            message = first_choice.get("message") if isinstance(first_choice, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            malformed_response = malformed_response or not isinstance(content, str) or not content.strip()
        cost = _usage_cost(response_json)
        rate_limit_kind = _rate_limit_kind(status_code, response_text, response_headers)
        if status_code == 429 and rate_limit_kind is not None:
            record_rate_limit_cooldown(
                provider=provider,
                model=model,
                role="throughput_probe",
                module="throughput_probe",
                run_id=lease_dir.parent.name,
                unit_id=f"{model}:c{step_concurrency}:r{request_index}",
                headers=response_headers,
                error=RuntimeError(response_text),
                lease_dir=lease_dir,
            )
        ok = 200 <= status_code < 300 and not malformed_response
        failure_kind = (
            "malformed_response"
            if malformed_response
            else _failure_kind(status_code, response_text, response_headers)
        )
        return {
            "ok": ok,
            "status_code": status_code,
            "rate_limited": rate_limit_kind is not None,
            "rate_limit_kind": rate_limit_kind,
            "failure_kind": None if ok else failure_kind,
            "latency_seconds": round(time.monotonic() - started, 4),
            "cost_usd": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "headers": {
                key: value
                for key, value in response_headers.items()
                if key.lower() in {"retry-after", "x-ratelimit-remaining", "x-ratelimit-reset", "x-ratelimit-limit"}
            },
            "rate_limit_delay_seconds": (
                rate_limit_delay_seconds(response_headers, default_seconds=0, max_seconds=300)
                if rate_limit_kind is not None
                else None
            ),
            "error": (
                None
                if ok
                else "HTTP 200 response was missing valid completion content"
                if malformed_response
                else sanitize_error_message(response_text)
            ),
            **timing_fields(),
        }
    except Exception as exc:
        rate_limited = is_rate_limit_error(exc)
        return {
            "ok": False,
            "status_code": status_code,
            "rate_limited": rate_limited,
            "rate_limit_kind": "rate_limit" if rate_limited else None,
            "failure_kind": "rate_limit" if rate_limited else _exception_failure_kind(exc),
            "latency_seconds": round(time.monotonic() - started, 4),
            "cost_usd": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "headers": response_headers,
            "rate_limit_delay_seconds": rate_limit_delay_seconds(error=exc, default_seconds=0, max_seconds=300) if rate_limited else None,
            "error": sanitize_error_message(exc),
            **timing_fields(),
        }


def summarize_step(concurrency: int, results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [
        float(item["latency_seconds"])
        for item in results
        if isinstance(item.get("latency_seconds"), (int, float))
    ]
    rate_limits = [item for item in results if item.get("rate_limited")]
    failures = [item for item in results if not item.get("ok")]
    non_rate_failures = [item for item in failures if not item.get("rate_limited")]

    def timing_summary(field: str) -> dict[str, float | None]:
        values = [
            float(item[field])
            for item in results
            if isinstance(item.get(field), (int, float))
        ]
        return {
            "median": round(median(values), 4) if values else None,
            "p95": percentile(values, 95),
            "max": round(max(values), 4) if values else None,
        }

    return {
        "concurrency": concurrency,
        "requests": len(results),
        "successes": sum(1 for item in results if item.get("ok")),
        "failures": len(failures),
        "rate_limits": len(rate_limits),
        "total_cost_usd": round(sum(float(item.get("cost_usd") or 0.0) for item in results), 6),
        "latency_seconds": {
            "min": round(min(latencies), 4) if latencies else None,
            "median": round(median(latencies), 4) if latencies else None,
            "p95": percentile(latencies, 95),
            "max": round(max(latencies), 4) if latencies else None,
        },
        "queue_wait_seconds": timing_summary("queue_wait_seconds"),
        "lease_hold_seconds": timing_summary("lease_hold_seconds"),
        "rate_limit_examples": rate_limits[:3],
        "failure_examples": non_rate_failures[:3],
    }


def _execute_probe(
    args: argparse.Namespace,
    *,
    api_key: str,
    key_info: dict[str, Any],
    chat_url: str,
    fake_provider: FakeOpenAIProvider | None = None,
) -> dict[str, Any]:
    run_id = args.run_id or f"throughput-probe-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    output_dir = Path(args.output_dir or (DEFAULT_OUTPUT_ROOT / run_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "THROUGHPUT_EVENTS.jsonl"
    lease_dir = output_dir / "_runtime" / "paid_call_leases"

    models = parse_models(args.models)
    steps = parse_steps(args.steps)
    requests_per_step = args.requests_per_step
    report: dict[str, Any] = {
        "schema_version": "benchmark-throughput-probe-v1",
        "mode": "local_fake" if fake_provider is not None else "paid_provider",
        "run_id": run_id,
        "created_at": utc_now(),
        "output_dir": str(output_dir),
        "chat_url": chat_url,
        "models": models,
        "steps": steps,
        "requests_per_step": requests_per_step,
        "budget_usd": args.budget_usd,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "key_info": key_info,
        "results": [],
        "total_cost_usd": 0.0,
    }
    atomic_write_json(output_dir / "THROUGHPUT_PROBE.json", report)

    total_cost = 0.0
    for model in models:
        safe_concurrency = 0
        model_result = {
            "model": model,
            "safe_concurrency": 0,
            "stopped_on_rate_limit": False,
            "steps": [],
        }
        for concurrency in steps:
            if total_cost >= args.budget_usd:
                model_result["stopped_reason"] = "budget_exhausted"
                break
            call_count = requests_per_step or concurrency
            set_paid_call_policy(
                concurrency,
                lease_dir=lease_dir,
                updated_by="throughput_probe",
            )
            paid_call_capacity = paid_call_capacity_report(lease_dir)
            effective_paid_call_limit = int(paid_call_capacity["effective_limit"])
            if fake_provider is not None:
                fake_provider.reset_stats()
            step_started = time.monotonic()
            limits = httpx.Limits(
                max_connections=concurrency,
                max_keepalive_connections=concurrency,
            )
            with httpx.Client(limits=limits) as http_client:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(
                            run_one_call,
                            model=model,
                            url=chat_url,
                            api_key=api_key,
                            prompt=args.prompt,
                            max_tokens=args.max_tokens,
                            timeout_seconds=args.timeout,
                            lease_dir=lease_dir,
                            step_concurrency=concurrency,
                            request_index=index,
                            http_client=http_client,
                        )
                        for index in range(call_count)
                    ]
                    results = [future.result() for future in as_completed(futures)]
            step_elapsed = max(0.000001, time.monotonic() - step_started)
            intervals = [
                (
                    float(item["_lease_started_monotonic"]),
                    float(item["_lease_finished_monotonic"]),
                )
                for item in results
                if isinstance(item.get("_lease_started_monotonic"), (int, float))
                and isinstance(item.get("_lease_finished_monotonic"), (int, float))
            ]
            for item in results:
                item.pop("_lease_started_monotonic", None)
                item.pop("_lease_finished_monotonic", None)
            step_summary = summarize_step(concurrency, results)
            step_summary["effective_paid_call_limit"] = effective_paid_call_limit
            step_summary["effective_paid_call_limit_source"] = paid_call_capacity[
                "effective_limit_source"
            ]
            step_summary["paid_call_capacity"] = paid_call_capacity
            lease_seconds = sum(float(item.get("lease_hold_seconds") or 0) for item in results)
            token_count = sum(
                int(item.get("prompt_tokens") or 0) + int(item.get("completion_tokens") or 0)
                for item in results
            )
            step_summary["elapsed_seconds"] = round(step_elapsed, 4)
            step_summary["slot_utilization"] = round(
                min(1.0, lease_seconds / (step_elapsed * effective_paid_call_limit)),
                4,
            )
            step_summary["steady_state_slot_utilization"] = _steady_state_utilization(
                intervals,
                effective_paid_call_limit,
            )
            step_summary["calls_per_minute"] = round(
                step_summary["successes"] * 60 / step_elapsed,
                2,
            )
            step_summary["tokens_per_minute"] = round(token_count * 60 / step_elapsed, 2)
            if fake_provider is not None:
                provider_stats = fake_provider.snapshot()
                elapsed = float(provider_stats["elapsed_seconds"])
                active_seconds = float(provider_stats["active_seconds"])
                step_summary["provider_max_active"] = int(provider_stats["max_active"])
                step_summary["provider_slot_utilization"] = round(
                    active_seconds / (elapsed * effective_paid_call_limit),
                    4,
                ) if elapsed > 0 else 0.0
            total_cost += step_summary["total_cost_usd"]
            model_result["steps"].append(step_summary)
            with events_path.open("a") as handle:
                handle.write(json.dumps(sanitize_ledger_value({
                    "timestamp": utc_now(),
                    "event": "probe_step_completed",
                    "model": model,
                    **step_summary,
                }), default=str) + "\n")
            if step_summary["rate_limits"]:
                model_result["stopped_on_rate_limit"] = True
                model_result["stopped_reason"] = "rate_limited"
                break
            if step_summary["failures"]:
                model_result["stopped_reason"] = "non_rate_limit_failure"
                break
            safe_concurrency = max(safe_concurrency, effective_paid_call_limit)
            model_result["safe_concurrency"] = safe_concurrency
            if args.pause_seconds > 0:
                time.sleep(args.pause_seconds)
        report["results"].append(model_result)
        report["total_cost_usd"] = round(total_cost, 6)
        report["updated_at"] = utc_now()
        atomic_write_json(output_dir / "THROUGHPUT_PROBE.json", report)
        if total_cost >= args.budget_usd:
            break

    return report


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
        raise ValueError("--budget-usd must be finite and greater than zero")
    if args.max_tokens <= 0 or args.requests_per_step < 0:
        raise ValueError("token and request counts must be positive/non-negative")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout must be finite and greater than zero")
    if not math.isfinite(args.pause_seconds) or args.pause_seconds < 0:
        raise ValueError("--pause-seconds must be finite and non-negative")
    if args.fake:
        with FakeOpenAIProvider(latency_seconds=args.fake_latency) as provider:
            return _execute_probe(
                args,
                api_key="local-fake-key",
                key_info={"available": True, "source": "local_fake", "limit_remaining": None},
                chat_url=provider.chat_url,
                fake_provider=provider,
            )

    if not args.confirm_paid_calls:
        raise SystemExit(
            "paid throughput probes require --confirm-paid-calls after reviewing "
            "the models, steps, and tracked-cost stop threshold"
        )
    require_credential_destination("OPENROUTER_API_KEY", args.chat_url)
    load_repo_env_files()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    key_info = sanitize_key_info(fetch_key_info(timeout=args.timeout))
    limit_remaining = key_info.get("limit_remaining")
    if isinstance(limit_remaining, (int, float)) and limit_remaining <= 0:
        raise SystemExit("OpenRouter key limit has no remaining credits")
    if isinstance(limit_remaining, (int, float)) and limit_remaining < args.budget_usd and not args.force:
        raise SystemExit(
            f"OpenRouter key has ${limit_remaining:.4f} remaining, below requested probe budget ${args.budget_usd:.4f}. "
            "Use --force to run anyway."
        )
    return _execute_probe(
        args,
        api_key=api_key,
        key_info=key_info,
        chat_url=args.chat_url,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure benchmark provider throughput with a local fake or paid endpoint.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated OpenRouter model ids.")
    parser.add_argument("--steps", default="1,2,3,4", help="Comma-separated concurrency steps.")
    parser.add_argument("--requests-per-step", type=int, default=0, help="Requests per step. Default: equal to concurrency.")
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=0.25,
        help=(
            "Tracked returned-usage cost stop threshold. This is not a provider "
            "hard cap; a step can exceed it when usage is absent or underreported."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Run even if key remaining is below requested budget.")
    parser.add_argument("--prompt", default="Reply with exactly: OK", help="Tiny probe prompt.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max completion tokens per probe call.")
    parser.add_argument("--timeout", type=float, default=60, help="Request timeout seconds.")
    parser.add_argument("--pause-seconds", type=float, default=1.0, help="Pause between concurrency steps.")
    parser.add_argument("--chat-url", default=DEFAULT_CHAT_URL, help="OpenRouter chat completions endpoint.")
    parser.add_argument("--fake", action="store_true", help="Use the deterministic local fake provider; makes no paid calls.")
    parser.add_argument(
        "--confirm-paid-calls",
        action="store_true",
        help="Confirm remote probe calls after reviewing the exact models and steps.",
    )
    parser.add_argument("--fake-latency", type=float, default=0.05, help="Fixed fake-provider response latency in seconds.")
    parser.add_argument("--run-id", help="Optional run id.")
    parser.add_argument("--output-dir", help="Optional output directory.")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_probe(args)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Throughput probe written to {report['output_dir']}")
        print(f"Total tracked cost: ${report['total_cost_usd']:.6f}")
        for result in report["results"]:
            marker = "rate-limited" if result.get("stopped_on_rate_limit") else "ok"
            print(f"- {result['model']}: safe_concurrency={result['safe_concurrency']} ({marker})")
            for step in result.get("steps") or []:
                print(
                    f"  requested={step['concurrency']} "
                    f"effective={step['effective_paid_call_limit']} "
                    f"source={step['effective_paid_call_limit_source']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
