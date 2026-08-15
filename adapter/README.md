# Reference adapter

The benchmark talks to systems under test through an OpenAI-compatible
`POST /v1/chat/completions` boundary. If your system already exposes that
contract, configure its URL in `suite_models.yaml` and skip this adapter.

Use this directory when you need either:

1. a local, deterministic endpoint for proving benchmark wiring; or
2. a thin translation boundary in front of a private or non-standard backend.

Private prompts, routing rules, credentials, traces, and customer data stay
behind the boundary. The public benchmark receives only a model id, assistant
text, finish/refusal signals, public usage counters, and safe error metadata.

## Free local proof

From the benchmark repository root:

```bash
# First run only: installs the locked adapter dependencies with the suite.
./scripts/verify-release-source
PYTHON_BIN=python3 ./scripts/bootstrap
test -e adapter/.env || (umask 077 && cp adapter/.env.example adapter/.env)
chmod 600 adapter/.env
./venv/bin/python adapter/server.py
```

Use that script entry point for unauthenticated loopback reference mode. Proxy
mode always requires `ADAPTER_INBOUND_API_KEY`, including on loopback, so a
webpage cannot trigger upstream calls through the local adapter. The chat route
also accepts only `application/json` requests and rejects browser `Origin`
headers. A Uvicorn import-string launch such as `uvicorn server:app` requires
inbound authentication because Uvicorn's own `--host` flag can otherwise
override the adapter's configured loopback bind.

The default configuration binds to `127.0.0.1:9999` and returns a deterministic
reference answer. It does not call a model provider.

In a second terminal:

```bash
./venv/bin/python adapter/smoke.py
```

The smoke verifies `/health`, `/v1/models`, and one complete chat response. It
refuses to send the chat request when the adapter is in upstream proxy mode
unless you explicitly add `--allow-proxy-call`, because that request may cost
money.

The adapter forwards the complete OpenAI message transcript unchanged. A
caller-supplied `conversation_id` is an optional correlation field and is also
forwarded unchanged; it is not evidence integrity, authentication, or a
durable adapter session. AITA continuity normally comes from stateless replay
of saved turns.

When the adapter requires inbound authentication, read the key from an
environment variable instead of placing it in process arguments:

```bash
./venv/bin/python adapter/smoke.py \
  --api-key-env LOCAL_OPENAI_COMPATIBLE_API_KEY
```

## Connect the benchmark

The public `suite_models.yaml` already contains this example:

```yaml
endpoints:
  local_openai_compatible:
    provider_api: openai_compatible
    openai_base_url: "http://127.0.0.1:9999/v1"
    chat_completions_url: "http://127.0.0.1:9999/v1/chat/completions"
    api_key_env: LOCAL_OPENAI_COMPATIBLE_API_KEY

models:
  local-openai-compatible:
    model_id: "local/example-model"
    label: "Local OpenAI-Compatible Endpoint"
    endpoint: local_openai_compatible
    max_parallel: 1
```

The preflight contract requires a non-empty key variable even when a loopback
service ignores authentication. For unauthenticated local development, put a
non-secret sentinel in the repository-root `.env`:

```bash
LOCAL_OPENAI_COMPATIBLE_API_KEY=local-development
```

Then verify the exact benchmark request shape against the free reference mode:

```bash
./venv/bin/python -m suite_tools.preflight_conditions \
  --group local_endpoint_smoke
```

Prepare a one-scenario run only after that probe passes. Scoring still uses the
configured judge and can incur provider cost even when generation uses the free
reference response.

## Proxy an OpenAI-compatible upstream

Set these values in `adapter/.env`:

```bash
UPSTREAM_OPENAI_BASE_URL=https://provider.example/v1
UPSTREAM_MODEL_ID=provider/private-model
UPSTREAM_API_KEY_ENV=PRIVATE_MODEL_API_KEY
```

`UPSTREAM_MODEL_ID` is optional. Blank means the adapter forwards the requested
model id. `UPSTREAM_CHAT_COMPLETIONS_URL` can override the derived
`<base-url>/chat/completions` URL.

The adapter normalizes the upstream response instead of blindly forwarding it.
A successful response must contain non-empty `choices[0].message.content` or an
explicit non-empty `message.refusal`. Refusal-only responses are preserved as
assistant text so the benchmark can treat the refusal as model behavior.
Malformed JSON, missing or null choices, unexplained empty content, timeouts,
transform failures, and non-200 responses become structured non-200 adapter
errors. The default error includes a SHA-256 digest of the raw upstream body,
not the body itself.

Adapter errors are also written out of band to the private path configured by
`ADAPTER_DIAGNOSTICS_PATH` (default
`adapter/results/adapter-diagnostics.jsonl`). These records are tagged
`adapter_claim`, contain no request messages, and do not add fields to the
OpenAI-compatible HTTP response. Set `ADAPTER_DIAGNOSTICS_INCLUDE_DETAIL=true`
only for a controlled local investigation; detail is redacted and truncated but
may still reflect private upstream context. Journals rotate at
`ADAPTER_DIAGNOSTICS_MAX_BYTES` and remain excluded from git and evidence
packages.

## Adapt a proprietary backend

Keep `server.py` unchanged and customize the explicit boundary in `backend.py`:

- `build_upstream_payload(request_body)` translates OpenAI messages into your
  private service's request JSON.
- `build_upstream_headers()` adds its authentication scheme.
- `parse_upstream_response(value)` extracts assistant text, finish/refusal
  signals, and public usage counters into `BackendCompletion`.

The complete, tested
[`examples/proprietary_json_backend.py`](examples/proprietary_json_backend.py)
shows a fictional service that accepts `{"prompt": ..., "history": ...}` and
returns `{"answer": ...}`. Copy its three hook functions into `backend.py`, add
any standard-library imports they need, then change the field names and
authentication header for your service. The result/error types are already
available in `backend.py`. In abbreviated form, the translation looks like
this:

```python
def build_upstream_payload(request_body):
    messages = request_body["messages"]
    return {
        "prompt": messages[-1]["content"],
        "history": messages[:-1],
    }


def parse_upstream_response(value):
    answer = value.get("answer") if isinstance(value, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise OpenAIContractError(
            "empty_upstream_content",
            "Upstream response contained no assistant text",
        )
    return BackendCompletion(content=answer)
```

Do not return exceptions, HTTP errors, or backend JSON as successful assistant
text. Raise `AdapterBackendError` or `OpenAIContractError`; the server will emit
a structured failure that the benchmark preserves rather than scores.

## Identity and reproducibility

The adapter's `EXPOSED_MODEL_ID` should match the model id declared in the suite
config. Put response-affecting private configuration into stable public hashes:

```yaml
models:
  my-system:
    model_id: "organization/my-system-v2"
    label: "My System v2"
    endpoint: local_openai_compatible
    condition_id: "my-system-v2-guardrails-2026-07"
    served_profile_hash: "sha256:<hash-of-private-prompt-and-routing-bundle>"
    provider_condition_hash: "sha256:<hash-of-full-served-condition>"
    condition_metadata:
      effort: "provider_default"
      version: "2026-07"
```

Publish the hashes and stable labels, not the private prompt or routing inputs.
Change the condition identity whenever weights, hidden prompts, safety layers,
tools, routing, or other response-affecting behavior changes.

## Exposure and logging safety

- The adapter binds to loopback by default.
- Non-loopback exposure is refused unless `ADAPTER_INBOUND_API_KEY` is
  configured, including in deterministic reference mode.
- `ADAPTER_MAX_REQUEST_BYTES` defaults to `1048576`; requests exceeding that
  limit are rejected before JSON validation.
- When inbound auth is enabled, set the same value as
  `LOCAL_OPENAI_COMPATIBLE_API_KEY` in the benchmark's root `.env`.
- `ADAPTER_DEBUG_UPSTREAM_ERRORS=false` is the safe default. Debug detail may
  contain private provider data and should never be enabled in published logs.
- Never commit `adapter/.env`, private overlays, backend traces, or real keys.

See [RUNBOOK.md section 2](../RUNBOOK.md#2-running-openai-compatible-endpoint-models)
for request fields, response fields, condition metadata, and the full run flow.
