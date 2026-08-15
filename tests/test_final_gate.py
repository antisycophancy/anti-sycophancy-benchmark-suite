"""Final-gate tests: F1 (exception-shaped model signals → BLOCKS) and
F2 (evidence envelope + billing carried to record_block).

Tests cover all three runners (aita, epis, sus) for:
- F1: guardrail-403 exception → BenchmarkProviderRefusal/ProviderRefusalError → record_block
- F1: record_outcome model_signal (guardrail_permission_denied) handled before retry path
- F1 residual: native ProviderRefusalError (Responses API / Anthropic) routed through
  classify_payload + executor: bounded_retry(1) → 2 calls + full envelope;
  Anthropic stop_reason=refusal → 1 call + provider=anthropic + terminal envelope.
- F2: _terminal_evidence propagation from classify_evidence through exception to record_block
- F2: stochastic 3-call case → 3 usage rows + billed_attempts=3 + full evidence on block
- signals Rule 9b: Responses API incomplete_details.reason=content_filter

Production-shaped exceptions: body via `.body` attr (SDK shape), raw_response via .raw_response.
Pattern: MagicMock client with side_effect, autouse lease-dir fixture (env var only).
"""
from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

import pytest

from suite_tools.provider_client import ProviderRefusalError


# ── Shared factories ──────────────────────────────────────────────────────────

def _make_choice(finish_reason=None, content=None, refusal=None, native_finish_reason=None):
    msg = MagicMock()
    msg.content = content
    msg.refusal = refusal
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.native_finish_reason = native_finish_reason
    choice.message = msg
    return choice


def _safety_resp():
    """Response with native_finish_reason=SAFETY on BOTH the response AND choice[0].

    - aita/epis T3 block path checks: ``resp.native_finish_reason or resp.choices[0].native_finish_reason``
    - sus T3 flat-dict path checks: ``_raw.get("native_finish_reason")`` which comes from
      ``_raw_openai_compatible_response(response)`` which reads ``choice.native_finish_reason``

    Both paths are covered by setting it in both places.
    """
    choice = _make_choice(finish_reason=None, content=None, native_finish_reason="SAFETY")
    resp = MagicMock()
    resp.choices = [choice]
    resp.native_finish_reason = "SAFETY"   # top-level for aita/epis
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 8
    resp.usage.completion_tokens = 1
    resp.usage.total_tokens = 9
    return resp


def _guardrail_body() -> dict:
    """OpenRouter-shaped permission_denied guardrail body (Rule 2).

    Rule 2 checks: ``error.metadata.error_type == "permission_denied"``
    and ``error.metadata.reasons`` (truthy).  Correct shape:
    {"error": {"metadata": {"error_type": "permission_denied", "reasons": ["guardrail"]}}}
    """
    return {
        "error": {
            "metadata": {
                "error_type": "permission_denied",
                "reasons": ["guardrail"],
            }
        }
    }


def _guardrail_sdk_exc() -> Exception:
    """SDK-shaped exception with guardrail body via .body attribute (aita/epis path)."""
    exc = RuntimeError("403 Permission Denied: guardrail blocked request")
    exc.status_code = 403
    exc.body = _guardrail_body()
    return exc


# ── provider_signals Rule 9b ──────────────────────────────────────────────────

def test_rule_9b_responses_incomplete_content_filter() -> None:
    """Plan 020 F1(a): status=incomplete + incomplete_details.reason=content_filter
    -> model_signal/content_filter, bounded_retry(1), provider=openai."""
    from suite_tools.provider_signals import classify_payload

    class _Holder:
        raw_response = {
            "id": "resp_abc",
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "output": [],
        }

    result = classify_payload(_Holder())
    assert result is not None, "Rule 9b should match"
    assert result["evidence_class"] == "model_signal"
    assert result["category"] == "content_filter"
    assert result["provider"] == "openai"
    assert result["retry_policy"]["kind"] == "bounded_retry"
    assert result["retry_policy"]["max_retries"] == 1


def test_rule_9b_max_tokens_not_matched() -> None:
    """max_output_tokens truncation must NOT match Rule 9b."""
    from suite_tools.provider_signals import classify_payload

    class _Holder:
        raw_response = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        }

    result = classify_payload(_Holder())
    assert result is None or result.get("category") != "content_filter"


# ── F1(b): flat-raw normalization ────────────────────────────────────────────

def test_sus_flat_shape_hits_rule5_bounded_retry() -> None:
    """F1(b): SUS T3 executor normalizes to flat raw dict; finish_reason at top-level
    hits Rule 5 (bounded_retry, provider=openrouter), not Rule 4b (terminal, deepseek)."""
    from suite_tools.content_block_policy import consult_content_block
    from suite_tools.provider_signals import classify_payload

    # Flat shape (aita path and new sus path) -> Rule 5
    flat = {"finish_reason": "content_filter", "native_finish_reason": None, "refusal": None}
    ev = consult_content_block(flat)
    assert ev is not None
    assert ev["retry_policy"]["kind"] == "bounded_retry"
    assert ev["provider"] == "openrouter"

    # Nested shape (old sus path) -> Rule 4b (terminal, deepseek)
    class _Nested:
        raw_response = {"choices": [{"finish_reason": "content_filter"}]}

    ev_nested = classify_payload(_Nested())
    assert ev_nested is not None
    assert ev_nested["retry_policy"]["kind"] == "terminal"
    assert ev_nested["provider"] == "deepseek"


# ── F1 aita: guardrail-403 -> ProviderRefusalError with full evidence ────────

class TestAitaGuardrailF1:
    """F1: SDK guardrail-403 is handled by api_call's record_outcome dispatch
    -> ProviderRefusalError (not RuntimeError), 1 call, _terminal_evidence carries
    provider/signal_source/category from classify_payload (not re-derived later)."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        monkeypatch.setattr("aita_bench.runner.time.sleep", lambda s: None)
        self._tmp = tmp_path

    def test_guardrail_403_raises_provider_refusal_not_runtime(self) -> None:
        """api_call: guardrail-403 SDK exc -> ProviderRefusalError, 1 call, full evidence."""
        from aita_bench import runner

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [_guardrail_sdk_exc()]

        with pytest.raises(ProviderRefusalError) as exc_info:
            runner.api_call(client, "test/model", [{"role": "user", "content": "hi"}], retries=3)

        e = exc_info.value
        assert client.chat.completions.create.call_count == 1, "terminal signal must not retry"

        ev = getattr(e, "_terminal_evidence", None)
        assert ev is not None, "_terminal_evidence not set on ProviderRefusalError"
        assert ev["evidence_class"] == "model_signal"
        assert ev["category"] == "guardrail_permission_denied"
        assert ev.get("provider") == "openrouter", f"provider missing: {ev}"
        assert ev.get("signal_source") is not None, f"signal_source missing: {ev}"
        assert getattr(e, "_billed_attempts", None) == 1

    def test_guardrail_403_run_conversation_record_block_full_evidence(
        self, monkeypatch
    ) -> None:
        """End-to-end: guardrail-403 -> run_conversation -> monitor.record_block with
        provider/signal_source from _terminal_evidence (not re-derived from bare exception)."""
        from aita_bench import runner
        from suite_tools.run_monitor import RunMonitor

        fake_client = MagicMock()
        fake_client.base_url = "https://openrouter.ai/api/v1"
        fake_client.chat.completions.create.side_effect = [_guardrail_sdk_exc()] * 10
        monkeypatch.setattr(runner, "make_client", lambda cfg: fake_client)

        monitor = RunMonitor(self._tmp, module="aita", stage="generation")
        blocks: list[dict] = []
        monitor.record_block = lambda **kw: blocks.append(kw)

        models = {
            "m": {
                "label": "M", "model_id": "test/m",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "fake", "max_parallel": 1,
            }
        }
        seeker_client = MagicMock()

        with pytest.raises(ProviderRefusalError):
            runner.run_conversation(
                "m", "a test post", 0, "side_a", self._tmp,
                seeker_client, models, monitor=monitor,
            )

        assert len(blocks) == 1, f"Expected 1 record_block call; got {len(blocks)}"
        ev = blocks[0]["evidence"]
        assert ev["evidence_class"] == "model_signal"
        assert ev["category"] == "guardrail_permission_denied"
        # F2: provider and signal_source come from _terminal_evidence, not re-derived
        assert ev.get("provider") == "openrouter", f"provider missing: {ev}"
        assert ev.get("signal_source") is not None, f"signal_source missing: {ev}"


# ── F1 epis: guardrail-403 -> ProviderRefusalError with full evidence ─────────

class TestEpisGuardrailF1:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        monkeypatch.setattr("epis_bench.runner.time.sleep", lambda s: None)

    def test_guardrail_403_raises_provider_refusal_not_runtime(self) -> None:
        """F1: epis api_call with guardrail-403 -> ProviderRefusalError, 1 call, full evidence."""
        from epis_bench import runner as erunner

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [_guardrail_sdk_exc()]

        with pytest.raises(ProviderRefusalError) as exc_info:
            erunner.api_call(
                client, "test/model", [{"role": "user", "content": "hi"}], retries=3
            )

        e = exc_info.value
        assert client.chat.completions.create.call_count == 1

        ev = getattr(e, "_terminal_evidence", None)
        assert ev is not None
        assert ev["evidence_class"] == "model_signal"
        assert ev["category"] == "guardrail_permission_denied"
        assert ev.get("provider") == "openrouter"
        assert ev.get("signal_source") is not None
        assert getattr(e, "_billed_attempts", None) == 1


# ── F1 sus: guardrail-403 -> BenchmarkProviderRefusal via send() dispatch ────

class TestSusGuardrailF1:
    """F1: call_provider wraps SDK guardrail-403 as BenchmarkApiError; runner's
    send() dispatch converts model_signal/record_outcome to BenchmarkProviderRefusal
    so finalize_provider_refusal fires and block_evidence carries full fields."""

    def test_guardrail_403_call_provider_raises_benchmark_api_error(
        self, monkeypatch, tmp_path
    ) -> None:
        """call_provider raises BenchmarkApiError(403) with raw_response = guardrail body."""
        import sus_bench.api as api
        from sus_bench.api import BenchmarkApiError

        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [_guardrail_sdk_exc()] * 10
        monkeypatch.setattr(api, "_openai_factory", lambda **k: fake_client)
        monkeypatch.setattr(
            api, "paid_call_lease",
            MagicMock(side_effect=lambda *a, **k: contextlib.nullcontext()),
        )

        with pytest.raises(BenchmarkApiError) as exc_info:
            api.call_provider(
                "test/model", [{"role": "user", "content": "hi"}], "fake",
                base_url="https://openrouter.ai/api/v1",
            )

        err = exc_info.value
        assert err.status_code == 403
        assert isinstance(err.raw_response, dict), f"raw_response not a dict: {err.raw_response}"
        assert "error" in err.raw_response

    def test_guardrail_403_run_scenario_block_evidence_full_fields(
        self, monkeypatch, tmp_path
    ) -> None:
        """End-to-end sus: run_scenario returns block_evidence with provider/signal_source
        from _terminal_evidence (not re-derived from bare exception)."""
        import sus_bench.runner as srunner
        from sus_bench.api import BenchmarkApiError

        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))

        # BenchmarkApiError(403) with correct guardrail body in raw_response.
        # _extract_body reads .raw_response -> classify_payload Rule 2 ->
        # model_signal/guardrail_permission_denied with provider=openrouter.
        api_err = BenchmarkApiError(403, "Permission denied: content policy")
        api_err.raw_response = _guardrail_body()
        api_err.usage_recorded = True

        monkeypatch.setattr(
            srunner, "call_openrouter",
            lambda *a, **k: (_ for _ in ()).throw(api_err),
        )
        monkeypatch.setattr(srunner, "time", MagicMock())

        scenario = {
            "id": "test-scenario",
            "name": "Test",
            "elicitation": {"prompt": "test prompt"},
            "detection": {
                "crisis_indicators": [],
                "compliance_indicators": [{"phrase": "sure"}],
            },
            "escalation": [],
        }
        model_config = {"id": "test/model", "label": "TestModel"}
        monitor = MagicMock()

        result = srunner.run_scenario(
            model_config, scenario, "fake_key", "test/analyzer",
            monitor=monitor,
        )

        assert result.get("score_state") == "excluded_provider_refusal", result.get("score_state")
        ev = result.get("block_evidence", {})
        assert ev.get("evidence_class") == "model_signal"
        assert ev.get("category") == "guardrail_permission_denied"
        assert ev.get("provider") == "openrouter", f"provider missing from evidence: {ev}"
        assert ev.get("signal_source") is not None, f"signal_source missing: {ev}"
        assert result.get("_block_billed_attempts", 1) == 1


# ── F2 stochastic 3-call billing tests ───────────────────────────────────────

class TestStochasticThreeCall:
    """F2: SAFETY (native_finish_reason=SAFETY via T3 content-block executor) ->
    3 paid calls (stochastic_retry(2) -> 2 retries -> terminalize on 3rd),
    billed_attempts=3, _terminal_evidence with stochastic=True and provider=google."""

    @pytest.fixture(autouse=True)
    def _env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        monkeypatch.setenv("BENCHMARK_OUTPUT_BUDGET_RETRIES", "2")

    def test_aita_stochastic_3_calls_billed_and_evidence(self, monkeypatch) -> None:
        """aita T3: SAFETY response x 3 -> billed_attempts=3, stochastic evidence,
        3 monitor.record_usage calls (one per successful-but-empty response)."""
        from aita_bench import runner

        monkeypatch.setattr("aita_bench.runner.time.sleep", lambda s: None)
        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [_safety_resp() for _ in range(10)]

        usage_rows: list = []
        monitor = MagicMock()
        monitor.record_usage = lambda *a, **k: usage_rows.append((a, k))

        with pytest.raises(ProviderRefusalError) as exc_info:
            runner.api_call(
                client, "google/gemini-3-flash", [{"role": "user", "content": "hi"}],
                retries=10, monitor=monitor,
            )

        e = exc_info.value
        assert client.chat.completions.create.call_count == 3
        assert getattr(e, "_billed_attempts", None) == 3
        ev = getattr(e, "_terminal_evidence", None)
        assert ev is not None
        assert ev.get("stochastic") is True
        assert ev.get("provider") == "google"
        assert ev.get("signal_source") is not None
        assert len(usage_rows) == 3, f"Expected 3 usage records; got {len(usage_rows)}"

    def test_epis_stochastic_3_calls_billed_and_evidence(self, monkeypatch) -> None:
        """epis T3: SAFETY response x 3 -> billed_attempts=3, stochastic evidence,
        3 monitor.record_usage calls."""
        from epis_bench import runner as erunner

        monkeypatch.setattr("epis_bench.runner.time.sleep", lambda s: None)
        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [_safety_resp() for _ in range(10)]

        usage_rows: list = []
        monitor = MagicMock()
        monitor.record_usage = lambda *a, **k: usage_rows.append((a, k))

        with pytest.raises(ProviderRefusalError) as exc_info:
            erunner.api_call(
                client, "google/gemini-3-flash", [{"role": "user", "content": "hi"}],
                retries=10, monitor=monitor,
            )

        e = exc_info.value
        assert client.chat.completions.create.call_count == 3
        assert getattr(e, "_billed_attempts", None) == 3
        ev = getattr(e, "_terminal_evidence", None)
        assert ev is not None
        assert ev.get("stochastic") is True
        assert ev.get("provider") == "google"
        assert ev.get("signal_source") is not None
        assert len(usage_rows) == 3

    def test_sus_stochastic_3_calls_billed_and_evidence(self, monkeypatch, tmp_path) -> None:
        """sus T3: SAFETY response-body path through content-block executor ->
        3 calls, _cost_tracker.record called 3 times, _billed_attempts=3,
        _terminal_evidence with stochastic=True/provider=google."""
        import sus_bench.api as api
        from sus_bench.api import BenchmarkProviderRefusal

        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [_safety_resp() for _ in range(10)]
        monkeypatch.setattr(api, "_openai_factory", lambda **k: fake_client)
        monkeypatch.setattr(
            api, "paid_call_lease",
            MagicMock(side_effect=lambda *a, **k: contextlib.nullcontext()),
        )
        # Patch _usage_to_dict to return non-empty so `if usage:` guard passes and
        # _cost_tracker.record is called for each attempt.
        monkeypatch.setattr(api, "_usage_to_dict", lambda u: {"prompt_tokens": 8, "total_tokens": 9})
        monkeypatch.setattr(api, "estimate_usage_cost", lambda model, u: dict(u, cost=0.001))

        cost_records: list = []
        monkeypatch.setattr(api._cost_tracker, "record", lambda *a, **k: cost_records.append(a))

        with pytest.raises(BenchmarkProviderRefusal) as exc_info:
            api.call_provider(
                "google/gemini-3", [{"role": "user", "content": "hi"}], "fake",
                base_url="https://openrouter.ai/api/v1",
                role="model_under_test",
            )

        e = exc_info.value
        assert fake_client.chat.completions.create.call_count == 3
        assert getattr(e, "_billed_attempts", None) == 3
        ev = getattr(e, "_terminal_evidence", None)
        assert ev is not None
        assert ev.get("stochastic") is True
        assert ev.get("provider") == "google"
        assert ev.get("signal_source") is not None
        assert len(cost_records) == 3, f"Expected 3 cost records; got {len(cost_records)}"


# ── F2 content-block evidence envelope (response-body path) ──────────────────

class TestContentBlockEvidenceEnvelope:
    """F2: T3 content-block path (finish_reason=content_filter in response body) carries
    _terminal_evidence with retry_policy/signal_source on the raised ProviderRefusalError.
    record_block uses _terminal_evidence rather than re-deriving from the bare exception."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        monkeypatch.setattr("aita_bench.runner.time.sleep", lambda s: None)
        self._tmp = tmp_path

    def _content_filter_resp(self):
        choice = _make_choice(finish_reason="content_filter", content="")
        resp = MagicMock()
        resp.choices = [choice]
        resp.native_finish_reason = None
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 8
        return resp

    def test_content_block_carries_terminal_evidence(self) -> None:
        """ProviderRefusalError raised by T3 path has _terminal_evidence with
        retry_policy and signal_source from classify_payload (Rule 5, bounded_retry(1))."""
        from aita_bench import runner

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [self._content_filter_resp() for _ in range(5)]

        with pytest.raises(ProviderRefusalError) as exc_info:
            runner.api_call(client, "test/model", [{"role": "user", "content": "hi"}], retries=5)

        e = exc_info.value
        ev = getattr(e, "_terminal_evidence", None)
        assert ev is not None, "_terminal_evidence not carried through content-block path"
        assert ev["evidence_class"] == "model_signal"
        assert ev["category"] == "content_filter"
        assert ev.get("retry_policy") is not None, f"retry_policy missing: {ev}"
        assert ev.get("signal_source") is not None, f"signal_source missing: {ev}"

    def test_record_block_uses_terminal_evidence_not_reclassified(
        self, monkeypatch
    ) -> None:
        """run_conversation -> record_block receives evidence with retry_policy and
        signal_source from _terminal_evidence (not re-derived from the bare exception)."""
        from aita_bench import runner
        from suite_tools.run_monitor import RunMonitor

        fake_client = MagicMock()
        fake_client.base_url = "https://openrouter.ai/api/v1"
        fake_client.chat.completions.create.side_effect = [
            self._content_filter_resp() for _ in range(5)
        ]
        monkeypatch.setattr(runner, "make_client", lambda cfg: fake_client)

        monitor = RunMonitor(self._tmp, module="aita", stage="generation")
        blocks: list[dict] = []
        monitor.record_block = lambda **kw: blocks.append(kw)

        models = {
            "m": {
                "label": "M", "model_id": "test/m",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "fake", "max_parallel": 1,
            }
        }
        seeker_client = MagicMock()

        with pytest.raises(ProviderRefusalError):
            runner.run_conversation(
                "m", "a test post", 0, "side_a", self._tmp,
                seeker_client, models, monitor=monitor,
            )

        assert len(blocks) == 1, f"Expected 1 record_block; got {len(blocks)}"
        ev = blocks[0]["evidence"]
        assert ev.get("signal_source") is not None, f"signal_source missing: {ev}"
        assert ev.get("retry_policy") is not None, f"retry_policy missing: {ev}"


# ── F1 residual: native ProviderRefusalError → classify_payload + executor ───

def _responses_incomplete_content_filter_pre() -> ProviderRefusalError:
    """Responses API ProviderRefusalError: status=incomplete + content_filter (Rule 9b).

    This is the shape raised by provider_client.py:1034 (Responses adapter) when the
    API returns status=incomplete with incomplete_details.reason=content_filter.
    classify_payload matches Rule 9b → bounded_retry(1)/openai/content_filter.
    """
    return ProviderRefusalError(
        "content blocked",
        raw_response={
            "id": "resp_abc",
            "status": "incomplete",
            "incomplete_details": {"reason": "content_filter"},
            "output": [],
        },
        usage={"prompt_tokens": 10, "total_tokens": 10},
    )


def _anthropic_native_refusal_pre() -> ProviderRefusalError:
    """Anthropic native ProviderRefusalError: stop_reason=refusal (no classify_payload rule).

    This is the shape raised by provider_client.py:725 (Anthropic adapter).
    classify_payload returns None (no matching rule) → synthesis path in catch site:
    provider=anthropic, signal_source=typed_refusal, retry_policy=terminal.
    """
    return ProviderRefusalError(
        "Anthropic native provider refusal; stop_reason=refusal",
        raw_response={"stop_reason": "refusal"},
        usage={"prompt_tokens": 12, "total_tokens": 12},
    )


class TestNativeProviderRefusalRetry:
    """F1 residual: native ProviderRefusalError catch sites now route raw_response
    through classify_payload + executor before terminalizing.

    Bounded_retry(1) → 2 paid calls + full evidence envelope.
    Anthropic terminal → 1 call + provider=anthropic + retry_policy_kind=terminal.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BENCHMARK_PAID_CALL_LEASE_DIR", str(tmp_path / "leases"))
        monkeypatch.setattr("aita_bench.runner.time.sleep", lambda s: None)
        monkeypatch.setattr("epis_bench.runner.time.sleep", lambda s: None)
        self._tmp = tmp_path

    def test_aita_responses_incomplete_content_filter_2_calls_full_envelope(
        self,
    ) -> None:
        """aita api_call: Responses API PRE (bounded_retry(1)/openai) → 2 calls,
        _billed_attempts=2, full evidence (provider=openai, signal_source, retry_policy)."""
        from aita_bench import runner

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [
            _responses_incomplete_content_filter_pre(),
            _responses_incomplete_content_filter_pre(),
        ]

        with pytest.raises(ProviderRefusalError) as exc_info:
            runner.api_call(
                client, "openai/gpt-4o", [{"role": "user", "content": "hi"}], retries=3
            )

        e = exc_info.value
        assert client.chat.completions.create.call_count == 2, (
            f"bounded_retry(1) must fire 2 calls; got {client.chat.completions.create.call_count}"
        )
        assert getattr(e, "_billed_attempts", None) == 2, (
            f"_billed_attempts must be 2; got {getattr(e, '_billed_attempts', None)}"
        )
        ev = getattr(e, "_terminal_evidence", None)
        assert ev is not None, "_terminal_evidence not set on native ProviderRefusalError"
        assert ev["evidence_class"] == "model_signal"
        assert ev["category"] == "content_filter"
        assert ev.get("provider") == "openai", f"provider missing/wrong: {ev}"
        assert ev.get("signal_source") is not None, f"signal_source missing: {ev}"
        assert (ev.get("retry_policy") or {}).get("kind") == "bounded_retry", (
            f"retry_policy missing/wrong: {ev}"
        )

    def test_epis_responses_incomplete_content_filter_2_calls_full_envelope(
        self,
    ) -> None:
        """epis api_call: Responses API PRE (bounded_retry(1)/openai) → 2 calls,
        _billed_attempts=2, full evidence (provider=openai, signal_source, retry_policy)."""
        from epis_bench import runner as erunner

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [
            _responses_incomplete_content_filter_pre(),
            _responses_incomplete_content_filter_pre(),
        ]

        with pytest.raises(ProviderRefusalError) as exc_info:
            erunner.api_call(
                client, "openai/gpt-4o", [{"role": "user", "content": "hi"}], retries=3
            )

        e = exc_info.value
        assert client.chat.completions.create.call_count == 2, (
            f"bounded_retry(1) must fire 2 calls; got {client.chat.completions.create.call_count}"
        )
        assert getattr(e, "_billed_attempts", None) == 2
        ev = getattr(e, "_terminal_evidence", None)
        assert ev is not None
        assert ev["evidence_class"] == "model_signal"
        assert ev["category"] == "content_filter"
        assert ev.get("provider") == "openai", f"provider missing/wrong: {ev}"
        assert ev.get("signal_source") is not None
        assert (ev.get("retry_policy") or {}).get("kind") == "bounded_retry"

    def test_aita_anthropic_native_refusal_1_call_terminal_envelope(self) -> None:
        """aita api_call: Anthropic native refusal (no classify_payload rule) → 1 call,
        synthesized terminal envelope carries provider=anthropic + signal_source + retry_policy=terminal."""
        from aita_bench import runner

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [_anthropic_native_refusal_pre()]

        with pytest.raises(ProviderRefusalError) as exc_info:
            runner.api_call(
                client, "anthropic/claude-opus-4", [{"role": "user", "content": "hi"}], retries=3
            )

        e = exc_info.value
        assert client.chat.completions.create.call_count == 1, (
            "terminal Anthropic refusal must not retry"
        )
        assert getattr(e, "_billed_attempts", None) == 1
        ev = getattr(e, "_terminal_evidence", None)
        assert ev is not None, "_terminal_evidence not set on Anthropic native refusal"
        assert ev.get("provider") == "anthropic", f"provider must be anthropic: {ev}"
        assert ev.get("signal_source") is not None, f"signal_source missing: {ev}"
        assert (ev.get("retry_policy") or {}).get("kind") == "terminal", (
            f"retry_policy must be terminal: {ev}"
        )
