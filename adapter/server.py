"""Provider-neutral OpenAI-compatible reference adapter.

Most benchmark users can point the suite directly at an OpenAI-compatible
endpoint. This server is a small optional proxy/smoke target: by default it
returns a deterministic reference response, and when configured with
`UPSTREAM_OPENAI_BASE_URL` it forwards chat completions to an upstream
OpenAI-compatible API.
"""

from __future__ import annotations

import json
import secrets
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

import backend
from backend import AdapterBackendError
from config import (
    ADAPTER_DIAGNOSTICS_INCLUDE_DETAIL,
    ADAPTER_DIAGNOSTICS_MAX_BYTES,
    ADAPTER_DIAGNOSTICS_PATH,
    ADAPTER_DEBUG_UPSTREAM_ERRORS,
    ADAPTER_HOST,
    ADAPTER_INBOUND_API_KEY,
    ADAPTER_MAX_REQUEST_BYTES,
    ADAPTER_PORT,
    EXPOSED_MODEL_ID,
    upstream_chat_completions_url,
)
from diagnostics import record_adapter_error
from model_routing import list_adapter_model_ids
from openai_contract import (
    OpenAIContractError,
    chat_completion_response,
    model_list_response,
    validate_chat_completion_request,
)

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
EXPOSURE_ERROR = "refusing non-loopback exposure without ADAPTER_INBOUND_API_KEY"
PROXY_AUTH_ERROR = "refusing upstream proxy mode without ADAPTER_INBOUND_API_KEY"


class AdapterAuthError(Exception):
    pass


def assert_safe_exposure(*, allow_unauthenticated_loopback: bool = True) -> None:
    if upstream_chat_completions_url() and not ADAPTER_INBOUND_API_KEY:
        raise RuntimeError(PROXY_AUTH_ERROR)
    if ADAPTER_HOST not in LOOPBACK_HOSTS and not ADAPTER_INBOUND_API_KEY:
        raise RuntimeError(EXPOSURE_ERROR)
    if not allow_unauthenticated_loopback and not ADAPTER_INBOUND_API_KEY:
        raise RuntimeError(
            "uvicorn import-string launch requires ADAPTER_INBOUND_API_KEY; "
            "use `python adapter/server.py` for unauthenticated loopback mode"
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Uvicorn import-string launches bypass the ``__main__`` block, so enforce
    # the exposure guard in application startup as well.
    assert_safe_exposure(
        allow_unauthenticated_loopback=bool(
            getattr(_app.state, "allow_unauthenticated_loopback", False)
        )
    )
    yield


app = FastAPI(
    title="Benchmark OpenAI-Compatible Reference Adapter",
    lifespan=lifespan,
)
# Only the controlled script entry point opts into local unauthenticated mode.
# An import-string launch cannot silently override the configured bind address.
app.state.allow_unauthenticated_loopback = False


@app.exception_handler(AdapterAuthError)
async def adapter_auth_error_handler(_request: Request, _exc: AdapterAuthError):
    return JSONResponse(
        {"error": "missing or invalid adapter API key", "code": "adapter_unauthorized"},
        status_code=401,
    )


async def require_adapter_api_key(request: Request) -> None:
    if not ADAPTER_INBOUND_API_KEY:
        return
    expected = f"Bearer {ADAPTER_INBOUND_API_KEY}"
    provided = request.headers.get("authorization", "")
    if not secrets.compare_digest(provided, expected):
        raise AdapterAuthError()


v1_router = APIRouter(prefix="/v1", dependencies=[Depends(require_adapter_api_key)])


@v1_router.post("/chat/completions")
async def chat_completions(request: Request):
    local_request_id = uuid.uuid4().hex
    if request.headers.get("origin"):
        return JSONResponse(
            {
                "error": "Browser-origin requests are not accepted",
                "code": "browser_origin_not_allowed",
            },
            status_code=403,
        )
    media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if media_type != "application/json":
        return JSONResponse(
            {
                "error": "Content-Type must be application/json",
                "code": "unsupported_media_type",
            },
            status_code=415,
        )
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > ADAPTER_MAX_REQUEST_BYTES:
                return request_too_large_response()
        except ValueError:
            pass
    chunks: list[bytes] = []
    total_bytes = 0
    async for chunk in request.stream():
        total_bytes += len(chunk)
        if total_bytes > ADAPTER_MAX_REQUEST_BYTES:
            return request_too_large_response()
        chunks.append(chunk)
    raw_bytes = b"".join(chunks)
    try:
        raw_body = json.loads(raw_bytes)
    except (TypeError, ValueError):
        return JSONResponse(
            {"error": "Request body must be valid JSON", "code": "invalid_json"},
            status_code=400,
        )
    try:
        body = validate_chat_completion_request(raw_body)
    except OpenAIContractError as exc:
        return JSONResponse(
            {"error": str(exc), "code": exc.code},
            status_code=400,
        )

    try:
        completion = await backend.complete_chat(body)
    except AdapterBackendError as exc:
        try:
            record_adapter_error(
                ADAPTER_DIAGNOSTICS_PATH,
                request_id=local_request_id,
                model=str(body.get("model") or EXPOSED_MODEL_ID),
                error=exc,
                include_detail=ADAPTER_DIAGNOSTICS_INCLUDE_DETAIL,
                max_bytes=ADAPTER_DIAGNOSTICS_MAX_BYTES,
            )
        except Exception as diagnostic_error:
            print(
                "Adapter diagnostic write failed: "
                f"{type(diagnostic_error).__name__}"
            )
        error_body: dict[str, object] = {
            "error": str(exc),
            "code": exc.code,
            "benchmark_action": "stop_run_preserve_artifacts",
            **{key: value for key, value in exc.context.items() if value is not None},
        }
        if ADAPTER_DEBUG_UPSTREAM_ERRORS and exc.detail is not None:
            error_body["detail"] = exc.detail
        digest = exc.context.get("raw_response_sha256")
        print(
            f"Adapter backend error: code={exc.code} status={exc.status_code}"
            + (f" raw_response_sha256={digest}" if digest else "")
        )
        if ADAPTER_DEBUG_UPSTREAM_ERRORS and exc.detail:
            print(f"Adapter backend debug detail: {exc.detail}")
        return JSONResponse(error_body, status_code=exc.status_code)

    return JSONResponse(
        chat_completion_response(
            model=EXPOSED_MODEL_ID,
            content=completion.content,
            completion_id_prefix="adapter",
            usage=completion.usage,
            finish_reason=completion.finish_reason,
            native_finish_reason=completion.native_finish_reason,
            refusal=completion.refusal,
        )
    )


def request_too_large_response() -> JSONResponse:
    return JSONResponse(
        {
            "error": "Request body exceeds adapter size limit",
            "code": "request_too_large",
        },
        status_code=413,
    )


@v1_router.get("/models")
async def list_models():
    mode = "proxy" if upstream_chat_completions_url() else "reference_response"
    return JSONResponse(
        model_list_response(list_adapter_model_ids(EXPOSED_MODEL_ID), owned_by="local"),
        headers={"X-Antisycophancy-Adapter-Mode": mode},
    )


@app.get("/health")
async def health():
    # Keep unauthenticated liveness intentionally free of configuration detail.
    return {"status": "ok"}


app.include_router(v1_router)


if __name__ == "__main__":
    import uvicorn

    assert_safe_exposure()
    app.state.allow_unauthenticated_loopback = True
    uvicorn.run(app, host=ADAPTER_HOST, port=ADAPTER_PORT)
