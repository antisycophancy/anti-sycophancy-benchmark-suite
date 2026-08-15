"""Provider API client for sus-bench.

All LLM calls go through this module. Provider-specific transport is delegated
to the shared suite provider adapters while SUS keeps its local cost, latency,
lease, and compatibility wrappers stable.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from suite_tools.paid_call_lease import (
    paid_call_lease,
    provider_from_base_url,
    record_rate_limit_cooldown,
)
from suite_tools.provider_client import (
    ProviderApiError,
    ProviderOutputBudgetExhaustedError,
    ProviderRefusalError,
    estimate_usage_cost,
    is_anthropic_messages_url,
    is_gemini_generate_content_url,
    is_openai_responses_url,
    make_provider_client,
    normalize_chat_payload_for_provider,
)
from suite_tools.evidence import MODEL_SIGNAL, classify_evidence
from suite_tools.request_receipts import record_effective_request
from suite_tools.run_monitor import nonnegative_finite_number, nonnegative_integer

# Default OpenRouter endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Default request timeout in seconds (300s needed for extended thinking models like GPT-5.4 Pro)
DEFAULT_TIMEOUT = 300

class CreditExhaustedError(Exception):
    """Raised when OpenRouter credits drop below the warning threshold."""
    pass


class BenchmarkApiError(RuntimeError):
    """Raised when a benchmark model call returns a non-benchmarkable response."""

    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text
        self.raw_response: dict[str, Any] | None = None
        super().__init__(f"HTTP {status_code}: {text[:300]}")


class BenchmarkProviderRefusal(ProviderRefusalError):
    """Raised when a provider classifier declines the model-under-test call.

    Re-parented onto :class:`ProviderRefusalError` (spec 015 §4 / plan 016 Task 5)
    so ``classify_evidence`` recognizes it as a MODEL_SIGNAL and SUS's
    ``finalize_provider_refusal`` catch still fires, while the extra
    ``model``/``role``/``latency_ms`` attributes are preserved.
    """

    def __init__(
        self,
        text: str,
        *,
        model: str,
        role: str,
        latency_ms: int,
        usage: dict[str, Any] | None = None,
        raw_response: dict[str, Any] | None = None,
        stop_details: dict[str, Any] | None = None,
    ):
        super().__init__(text, raw_response=raw_response, usage=usage)
        self.model, self.role, self.latency_ms = model, role, latency_ms
        if stop_details:
            self.stop_details = stop_details
        self.stop_reason = self.stop_reason or "refusal"


def _lease_context() -> dict[str, str]:
    """Return scheduler-provided context for paid-call lease attribution."""
    context = {
        "module": os.environ.get("BENCHMARK_MODULE"),
        "run_id": os.environ.get("BENCHMARK_RUN_ID"),
        "output_dir": os.environ.get("BENCHMARK_OUTPUT_DIR"),
        "contract_path": os.environ.get("BENCHMARK_CONTRACT_PATH"),
    }
    return {key: value for key, value in context.items() if value}


class CostTracker:
    """Tracks cumulative API costs and monitors remaining OpenRouter credits."""

    def __init__(self):
        self._lock = threading.RLock()
        self.total_cost = 0.0
        self.calls = 0
        self.cost_by_model: dict[str, float] = {}
        self.cost_by_role: dict[str, float] = {}  # "model_under_test", "analyzer", "judge"
        self.reported_cost = 0.0
        self.estimated_cost = 0.0
        self.cost_by_source: dict[str, float] = {}
        self.usage_anomaly_count = 0
        self.invalid_usage_fields: dict[str, int] = {}
        self.tokens_in = 0
        self.tokens_out = 0
        self.thinking_tokens_out = 0
        self.unknown_cost_calls = 0
        self.unknown_cost_by_model: dict[str, int] = {}
        self._api_key: str | None = None
        self._credit_warning_threshold = 1.0  # warn when below $1
        self._credit_stop_threshold = 0.25     # stop when below $0.25
        self._calls_since_check = 0
        self._check_every_n_calls = 10         # check balance every N calls
        self._last_credit_remaining: float | None = None

    def set_api_key(self, api_key: str):
        """Set the API key for credit balance checks."""
        with self._lock:
            self._api_key = api_key

    def record(self, model: str, usage: dict, role: str = "unknown"):
        with self._lock:
            raw_estimated_cost = usage.get("estimated_cost")
            raw_cost = usage.get("cost")
            reported_value, reported_valid = nonnegative_finite_number(raw_cost)
            estimated_value, estimated_valid = nonnegative_finite_number(raw_estimated_cost)
            source_hint = str(usage.get("cost_source") or "").lower()
            cost_field_is_estimate = reported_valid and (
                "estimate" in source_hint or source_hint == "pricing_snapshot"
            )
            has_reported_cost = reported_valid and not cost_field_is_estimate
            has_estimated_cost = not has_reported_cost and (
                estimated_valid or cost_field_is_estimate
            )
            has_cost = has_reported_cost or has_estimated_cost
            cost = (
                reported_value
                if has_reported_cost
                else estimated_value
                if estimated_valid
                else reported_value
            )
            invalid_fields = []
            if raw_cost is not None and not reported_valid:
                invalid_fields.append("cost")
            if raw_estimated_cost is not None and not estimated_valid:
                invalid_fields.append("estimated_cost")
            if has_reported_cost and estimated_valid:
                invalid_fields.append("conflicting_cost_sources")
            prompt_tokens, prompt_valid = nonnegative_integer(usage.get("prompt_tokens"))
            completion_tokens, completion_valid = nonnegative_integer(
                usage.get("completion_tokens")
            )
            if usage.get("prompt_tokens") is not None and not prompt_valid:
                invalid_fields.append("prompt_tokens")
            if usage.get("completion_tokens") is not None and not completion_valid:
                invalid_fields.append("completion_tokens")
            self.total_cost += cost
            self.calls += 1
            self.cost_by_model[model] = self.cost_by_model.get(model, 0) + cost
            self.cost_by_role[role] = self.cost_by_role.get(role, 0) + cost
            self.reported_cost += cost if has_reported_cost else 0
            self.estimated_cost += cost if has_estimated_cost else 0
            cost_source = str(
                usage.get("cost_source")
                or (
                    "provider_reported" if has_reported_cost else
                    "pricing_snapshot" if has_estimated_cost else
                    "unknown"
                )
            )
            self.cost_by_source[cost_source] = self.cost_by_source.get(cost_source, 0) + cost
            if not has_cost:
                self.unknown_cost_calls += 1
                self.unknown_cost_by_model[model] = self.unknown_cost_by_model.get(model, 0) + 1
            self.tokens_in += prompt_tokens
            self.tokens_out += completion_tokens
            completion_details = usage.get("completion_tokens_details")
            if not isinstance(completion_details, dict):
                completion_details = {}
            raw_thinking_tokens = (
                usage.get("thoughts_tokens", 0)
                or completion_details.get("reasoning_tokens", 0)
                or usage.get("reasoning_tokens", 0)
                or 0
            )
            thinking_tokens, thinking_valid = nonnegative_integer(raw_thinking_tokens)
            if raw_thinking_tokens is not None and not thinking_valid:
                invalid_fields.append("thinking_tokens")
            self.thinking_tokens_out += thinking_tokens
            self.usage_anomaly_count += len(invalid_fields)
            for field in invalid_fields:
                self.invalid_usage_fields[field] = self.invalid_usage_fields.get(field, 0) + 1
            self._calls_since_check += 1

    def check_credit_if_due(self) -> float | None:
        """Stop before the next provider call, never after consuming a response."""
        with self._lock:
            if self._api_key and self._calls_since_check >= self._check_every_n_calls:
                self._check_credit_balance()
            return self._last_credit_remaining

    def _check_credit_balance(self):
        """Query OpenRouter for remaining account credit balance.

        Uses /api/v1/credits for actual account balance (not key limit cap).
        Falls back to /api/v1/auth/key if credits endpoint fails.
        """
        self._calls_since_check = 0
        try:
            from rich.console import Console
            console = Console(stderr=True)

            # Primary: actual account credits
            resp = httpx.get(
                "https://openrouter.ai/api/v1/credits",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                total_credits = data.get("total_credits")
                total_usage = data.get("total_usage")
                if total_credits is not None and total_usage is not None:
                    remaining = total_credits - total_usage
                    self._last_credit_remaining = remaining
                    if remaining <= self._credit_stop_threshold:
                        console.print(
                            f"\n[bold red]CREDIT EXHAUSTED: ${remaining:.2f} remaining "
                            f"(credits: ${total_credits:.2f}, used: ${total_usage:.2f}). "
                            f"Stopping.[/bold red]"
                        )
                        raise CreditExhaustedError(
                            f"OpenRouter account balance: ${remaining:.2f} "
                            f"(below ${self._credit_stop_threshold:.2f} stop threshold)"
                        )
                    elif remaining <= self._credit_warning_threshold:
                        console.print(
                            f"\n[yellow]CREDIT WARNING: ${remaining:.2f} remaining "
                            f"(credits: ${total_credits:.2f}, used: ${total_usage:.2f})[/yellow]"
                        )
                    return

            # Fallback: key limit (less accurate but better than nothing)
            resp = httpx.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                limit = data.get("limit")
                usage = data.get("usage")
                if limit is not None and usage is not None:
                    remaining = limit - usage
                    self._last_credit_remaining = remaining
        except CreditExhaustedError:
            raise
        except Exception:
            pass  # Don't crash the benchmark over a balance check failure

    def check_credit_now(self) -> float | None:
        """Force an immediate credit check. Returns remaining balance or None."""
        with self._lock:
            if self._api_key:
                self._check_credit_balance()
            return self._last_credit_remaining

    def summary(self) -> dict:
        with self._lock:
            result = {
                "total_cost_usd": round(self.total_cost, 8),
                "total_calls": self.calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "thinking_tokens_out": self.thinking_tokens_out,
                "cost_by_model": {k: round(v, 8) for k, v in self.cost_by_model.items()},
                "cost_by_role": {k: round(v, 8) for k, v in self.cost_by_role.items()},
                "reported_cost_usd": round(self.reported_cost, 8),
                "estimated_cost_usd": round(self.estimated_cost, 8),
                "cost_by_source": {
                    k: round(v, 8) for k, v in sorted(self.cost_by_source.items())
                },
                "usage_anomaly_count": self.usage_anomaly_count,
                "invalid_usage_fields": dict(sorted(self.invalid_usage_fields.items())),
                "unknown_cost_calls": self.unknown_cost_calls,
                "unknown_cost_by_model": dict(self.unknown_cost_by_model),
            }
            if self._last_credit_remaining is not None:
                result["credit_remaining_usd"] = round(self._last_credit_remaining, 2)
            return result


# Global cost tracker — reset per benchmark run
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    return _cost_tracker


def reset_cost_tracker():
    global _cost_tracker
    _cost_tracker = CostTracker()


def _openai_compatible_empty_text_error(data: dict) -> BenchmarkApiError:
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    finish_reason = choice.get("finish_reason") or "unknown"
    native_finish_reason = data.get("native_finish_reason")
    msg_dict = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    refusal = msg_dict.get("refusal")
    message = (
        "OpenAI-compatible response contained empty message content; "
        f"finish_reason={finish_reason}"
    )
    err = BenchmarkApiError(502, message)
    err.raw_response = {
        "finish_reason": finish_reason,
        "native_finish_reason": native_finish_reason,
        "refusal": refusal,
    }
    return err


def _provider_api_for_url(url: str) -> str:
    if is_anthropic_messages_url(url):
        return "anthropic_messages"
    if is_gemini_generate_content_url(url):
        return "gemini_generate_content"
    if is_openai_responses_url(url):
        return "openai_responses"
    return "openai_compatible"


def _openai_sdk_base_url(url: str) -> str:
    suffix = "/chat/completions"
    stripped = url.rstrip("/")
    if stripped.lower().endswith(suffix):
        return stripped[: -len(suffix)]
    return url


def _usage_to_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
        return data if isinstance(data, dict) else {}

    result: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
        "cost_source",
        "thoughts_tokens",
    ):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    return result


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def _raw_openai_compatible_response(response: Any) -> dict:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return {"choices": []}
    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    native_finish_reason = getattr(choice, "native_finish_reason", None)
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    refusal = getattr(message, "refusal", None)
    msg_dict: dict = {"content": content}
    if refusal is not None:
        msg_dict["refusal"] = refusal
    raw: dict = {
        "choices": [
            {
                "message": msg_dict,
                "finish_reason": finish_reason,
            }
        ]
    }
    if native_finish_reason is not None:
        raw["native_finish_reason"] = native_finish_reason
    return raw


def _error_status_code(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _error_text(exc: BaseException) -> str:
    text = getattr(exc, "text", None)
    if isinstance(text, str):
        return text
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", None)
    if isinstance(response_text, str):
        return response_text
    return str(exc)


def _openai_factory(**kwargs):
    return OpenAI(**kwargs)


def _record_error_usage(
    model: str,
    role: str,
    exc: BaseException,
    *,
    monitor: Any = None,
    provider: str = "unknown",
) -> None:
    usage = _usage_to_dict(getattr(exc, "usage", None))
    usage = estimate_usage_cost(model, usage) if usage else {}
    _cost_tracker.record(model, usage, role=role)
    record_usage = getattr(monitor, "record_usage", None)
    if callable(record_usage):
        record_usage(
            model,
            usage,
            role=role,
            provider=provider,
            allow_empty=True,
        )


def _configured_output_budget_retries() -> int:
    """Bounded retries for stochastic output-budget exhaustion / stochastic
    MODEL_SIGNALs (default 2).

    These are stochastic runaway/blocked outcomes that usually resolve on replay,
    so they are retried this many extra times before terminalizing. ``0`` restores
    immediate-terminal behavior. Independent of the transient-error retry budget.
    """
    raw = os.environ.get("BENCHMARK_OUTPUT_BUDGET_RETRIES", "2")
    try:
        retries = int(raw)
    except ValueError as exc:
        raise ValueError("BENCHMARK_OUTPUT_BUDGET_RETRIES must be a non-negative integer") from exc
    if retries < 0:
        raise ValueError("BENCHMARK_OUTPUT_BUDGET_RETRIES must be a non-negative integer")
    return retries


def call_provider(
    model: str,
    messages: list[dict],
    api_key: str,
    *,
    base_url: str | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    request_options: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    role: str = "unknown",
    monitor: Any = None,
    request_context: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Send messages to a model through the configured provider adapter.

    Args:
        model: Model identifier (e.g., "anthropic/claude-sonnet-4.6").
        messages: Chat messages in OpenAI format.
        api_key: API key for authentication.
        base_url: Override the endpoint URL. Defaults to OpenRouter.
        temperature: Sampling temperature. None = provider default.
        reasoning_effort: Reasoning effort level (none/minimal/low/medium/high/xhigh).
            Applies to OpenRouter/OpenAI-compatible reasoning payloads. Native
            provider thinking controls should be passed through request_options.
        request_options: Additional provider request fields for this model condition
            (for example OpenRouter Opus verbosity controls). These are applied
            to the target payload and may override ``reasoning_effort``.
        timeout: Request timeout in seconds.
        role: Cost tracking label ("model_under_test", "analyzer", "judge").

    Returns:
        Tuple of (response_text, latency_ms).
    """
    url = base_url or OPENROUTER_URL
    provider_api = _provider_api_for_url(url)
    client_base_url = _openai_sdk_base_url(url) if provider_api == "openai_compatible" else url
    client = make_provider_client(
        {
            "api_key": api_key,
            "base_url": client_base_url,
            "provider_api": provider_api,
        },
        openai_factory=_openai_factory,
    )

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "timeout": timeout,
    }
    if provider_api != "openai_compatible":
        create_kwargs["record_rate_limit_cooldown"] = False
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    extra_body: dict[str, Any] = {}
    if reasoning_effort is not None and provider_api == "openai_compatible":
        extra_body["reasoning"] = {"effort": reasoning_effort}
    if reasoning_effort is not None and provider_api == "openai_responses":
        # The Responses client maps reasoning_effort -> reasoning.effort.
        create_kwargs["reasoning_effort"] = reasoning_effort
    if request_options:
        if not isinstance(request_options, dict):
            raise ValueError("request_options must be a mapping")
        extra_body.update(request_options)
    if provider_api == "openai_compatible" and "max_tokens" in extra_body:
        create_kwargs["max_tokens"] = extra_body.pop("max_tokens")
    if provider_api == "openai_compatible" and "max_completion_tokens" in extra_body:
        create_kwargs["max_completion_tokens"] = extra_body.pop("max_completion_tokens")
    if extra_body:
        create_kwargs["extra_body"] = extra_body
    normalize_chat_payload_for_provider(create_kwargs, base_url=client_base_url)

    provider = provider_from_base_url(url)
    lease_context = _lease_context()

    # Single-owner billing (plan 016 Task 5 / M5): THIS layer records usage for
    # every attempt exactly once and stamps ``usage_recorded=True`` on every
    # exception it re-raises, so outer runner catches never double-bill. The
    # budget/stochastic bounded-retry loop lives HERE (before the
    # ProviderApiError->BenchmarkApiError conversion) so the raw typed error stays
    # classifiable by ``classify_evidence``.
    #
    # T3/D8: content-block policy executor owns a THIRD independent retry counter
    # (executor._signal_attempts) for signals detected in successful but empty
    # responses. It does not consume the budget_attempts counter, and the
    # content-block retry loop calls continue without incrementing budget_attempts.
    # Deferred import: keeps module-level import clean for the nested-import test.
    from suite_tools.content_block_policy import ContentBlockPolicyExecutor, consult_content_block  # noqa: PLC0415
    from suite_tools.call_diagnostics import (  # noqa: PLC0415
        begin_provider_attempt,
        close_error_best_effort,
        close_success_best_effort,
    )
    output_budget_retries = _configured_output_budget_retries()
    budget_attempts = 0
    executor = ContentBlockPolicyExecutor()
    provider_attempts = 0
    while True:
        _cost_tracker.check_credit_if_due()
        start = time.time()
        provider_invocation_started = False
        try:
            provider_attempts += 1
            receipt: dict[str, Any] = {}
            if monitor is not None:
                receipt_context = dict(request_context or {})
                receipt_context.setdefault("provider", provider)
                receipt_context.setdefault("provider_api", provider_api)
                receipt = record_effective_request(
                    monitor,
                    create_kwargs,
                    base_url=client_base_url,
                    role=role,
                    call_attempt=provider_attempts,
                    **receipt_context,
                )
            diagnostic = begin_provider_attempt(
                monitor=monitor,
                output_dir=lease_context.get("output_dir"),
                module=lease_context.get("module", "sus"),
                role=role,
                model=model,
                provider=provider,
                provider_api=provider_api,
                context={**receipt, **dict(request_context or {})},
            )
            try:
                with paid_call_lease(provider=provider, model=model, role=role, **lease_context):
                    diagnostic.mark_provider_invocation_started()
                    provider_invocation_started = True
                    response = client.chat.completions.create(**create_kwargs)
            except Exception as exc:
                close_error_best_effort(diagnostic, exc, monitor)
                raise
            close_success_best_effort(diagnostic, response, monitor)
        except (ProviderRefusalError, ProviderApiError) as exc:
            latency = int((time.time() - start) * 1000)
            evidence = classify_evidence(exc)
            is_budget = isinstance(exc, ProviderOutputBudgetExhaustedError)
            is_stochastic = (
                evidence.get("evidence_class") == MODEL_SIGNAL and evidence.get("stochastic")
            )
            if is_budget or is_stochastic:
                # Bounded-retryable stochastic outcome: bill this attempt once, then
                # retry until the budget is spent before terminalizing.
                _record_error_usage(
                    model, role, exc, monitor=monitor, provider=provider
                )
                if budget_attempts < output_budget_retries:
                    budget_attempts += 1
                    continue
                if is_budget:
                    # Terminal: re-raise as-is so run_scenario's
                    # finalize_output_budget_exhausted catch fires.
                    exc.usage_recorded = True
                    raise
                # Terminal stochastic MODEL_SIGNAL -> a BenchmarkProviderRefusal so the
                # existing finalize_provider_refusal catch fires and the block records
                # the stochastic category.
                refusal = BenchmarkProviderRefusal(
                    exc.text,
                    model=model,
                    role=role,
                    latency_ms=latency,
                    usage=exc.usage,
                    raw_response={"finish_reason": evidence.get("category"), "stochastic_exhausted": True},
                )
                refusal.usage_recorded = True
                # F2: carry full evidence envelope + true billed count.
                refusal._terminal_evidence = evidence
                refusal._billed_attempts = budget_attempts + 1
                raise refusal from exc
            if isinstance(exc, ProviderRefusalError):
                _record_error_usage(
                    model, role, exc, monitor=monitor, provider=provider
                )
                # F1 residual + F2: classify_payload on exc.raw_response yields the
                # full evidence record (provider/signal_source/retry_policy); where no
                # rule matches (e.g. Anthropic stop_reason=refusal), synthesize a
                # terminal envelope so finalize_provider_refusal never sees bare
                # {model_signal, refusal}.  Bounded-retry policies (Rule 9b) get their
                # re-attempt(s) via the shared executor before terminalizing.
                from suite_tools.provider_signals import classify_payload as _cp  # noqa: PLC0415
                _classified = _cp(exc)
                if _classified is not None:
                    _ev = _classified
                else:
                    # No classify_payload rule matched (e.g. Anthropic stop_reason=refusal):
                    # synthesize terminal envelope from classify_evidence result.
                    _ev = dict(evidence)  # has {evidence_class, category}
                    _rr = getattr(exc, "raw_response", None) or {}
                    if isinstance(_rr, dict) and _rr.get("stop_reason") == "refusal":
                        _ev["provider"] = "anthropic"
                    _ev.setdefault("signal_source", "typed_refusal")
                    _ev.setdefault("retry_policy", {"kind": "terminal", "max_retries": 0})
                _rp_kind = (_ev.get("retry_policy") or {}).get("kind", "terminal")
                if _rp_kind == "bounded_retry" and executor.decide(_ev) == "continue":
                    # Retry allowed; usage already billed above, loop back.
                    continue
                # Terminalizing: construct BenchmarkProviderRefusal with full envelope.
                refusal = BenchmarkProviderRefusal(
                    exc.text,
                    model=model,
                    role=role,
                    latency_ms=latency,
                    usage=exc.usage,
                    raw_response=exc.raw_response,
                    stop_details=exc.stop_details,
                )
                refusal.usage_recorded = True
                refusal._terminal_evidence = _ev
                refusal._billed_attempts = executor.billed_attempt_count()
                raise refusal from exc
            _record_error_usage(
                model, role, exc, monitor=monitor, provider=provider
            )
            if exc.status_code == 429 and evidence.get("category") != "billing":
                record_rate_limit_cooldown(
                    provider=provider,
                    model=model,
                    role=role,
                    module=lease_context.get("module"),
                    run_id=lease_context.get("run_id"),
                    headers=dict(getattr(exc, "headers", {}) or {}),
                    error=BenchmarkApiError(exc.status_code, exc.text[:500]),
                )
            err = BenchmarkApiError(exc.status_code, exc.text[:500])
            err.raw_response = getattr(exc, "raw_response", None)
            err.usage_recorded = True
            raise err from exc
        except Exception as exc:
            if provider_invocation_started and not getattr(exc, "usage_recorded", False):
                _record_error_usage(
                    model,
                    role,
                    exc,
                    monitor=monitor,
                    provider=provider,
                )
            status_code = _error_status_code(exc)
            if status_code is not None:
                text = _error_text(exc)[:500]
                if (
                    status_code == 429
                    and classify_evidence(exc).get("category") != "billing"
                ):
                    record_rate_limit_cooldown(
                        provider=provider,
                        model=model,
                        role=role,
                        module=lease_context.get("module"),
                        run_id=lease_context.get("run_id"),
                        error=BenchmarkApiError(status_code, text),
                    )
                err = BenchmarkApiError(status_code, text)
                try:
                    from suite_tools.provider_client import extract_raw_response as _erx
                    err.raw_response = _erx(exc)
                except ImportError:
                    err.raw_response = None
                err.usage_recorded = True
                raise err from exc
            exc.usage_recorded = True
            raise

        # ── Successful API call ─────────────────────────────────────────────
        # Compute latency and record usage HERE (inside the loop) so that a
        # content-block retry (continue below) bills and times each attempt
        # independently before going back for the next paid call.
        latency = int((time.time() - start) * 1000)

        usage = _usage_to_dict(getattr(response, "usage", None))
        usage = estimate_usage_cost(model, usage) if usage else {}
        _cost_tracker.record(model, usage, role=role)
        record_usage = getattr(monitor, "record_usage", None)
        if callable(record_usage):
            record_usage(
                model,
                usage,
                role=role,
                provider=provider,
                allow_empty=True,
            )

        content = _response_text(response)
        if not content:
            if provider_api == "openai_compatible":
                # T3/D9: consult content-block policy on the RAW response BEFORE
                # raising any exception.
                # F1(b): normalize to FLAT shape (finish_reason at top-level, NOT
                # nested in choices[0]) so the classifier always hits Rule 5
                # (bounded_retry(1), provider="openrouter") rather than Rule 4b
                # (terminal, provider="deepseek") for OpenAI-compatible paths.
                # Keeps _raw (nested) for the error-path helper that expects choices.
                _raw = _raw_openai_compatible_response(response)
                _choices = _raw.get("choices") or []
                _c0 = _choices[0] if _choices else {}
                _flat = {
                    "finish_reason": _c0.get("finish_reason"),
                    "native_finish_reason": _raw.get("native_finish_reason"),
                    "refusal": (_c0.get("message") or {}).get("refusal"),
                }
                _block_ev = consult_content_block(_flat)
                if _block_ev is not None:
                    if executor.decide(_block_ev) == "continue":
                        # Bound not yet exhausted — usage already billed above;
                        # loop back for another paid attempt without touching
                        # budget_attempts (content-block counter is independent).
                        continue
                    # Bound exhausted (or terminal) → route through the EXISTING
                    # BenchmarkProviderRefusal conversion path (D9) so
                    # run_scenario's finalize_provider_refusal catch fires.
                    # usage_recorded=True: billing already done by _cost_tracker.
                    _err = _openai_compatible_empty_text_error(_raw)
                    refusal = BenchmarkProviderRefusal(
                        _err.text,
                        model=model,
                        role=role,
                        latency_ms=latency,
                        usage=usage or {},
                        raw_response=_err.raw_response,
                    )
                    refusal.usage_recorded = True
                    refusal._billed_attempts = executor.billed_attempt_count()
                    # F2: carry full evidence so finalize_provider_refusal uses it.
                    refusal._terminal_evidence = _block_ev
                    raise refusal
                # Genuinely unexplained empty (no signal matched) → keep the
                # existing 502 / retry path in run_scenario unchanged.
                raise _openai_compatible_empty_text_error(_raw)
            raise BenchmarkApiError(502, f"{provider_api} response contained empty message content")
        break
    return content, latency


def call_openrouter(*args, **kwargs) -> tuple[str, int]:
    """Backward-compatible alias for the historical SUS provider helper."""
    return call_provider(*args, **kwargs)


def parse_llm_json(text: str) -> dict:
    """Parse JSON from an LLM response, stripping markdown code fences.

    Handles common LLM output patterns:
      - Raw JSON
      - ```json ... ```
      - ``` ... ```
      - Extra text before/after JSON

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict.

    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted.
    """
    cleaned = text.strip()

    def reject_nonstandard_constant(token: str):
        raise json.JSONDecodeError(
            f"Non-standard numeric constant is not valid benchmark JSON: {token}",
            token,
            0,
        )

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(cleaned, parse_constant=reject_nonstandard_constant)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        return json.loads(match.group(), parse_constant=reject_nonstandard_constant)

    raise json.JSONDecodeError("No valid JSON found", text, 0)
