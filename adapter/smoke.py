"""No-cost contract smoke for the provider-neutral reference adapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx


def _service_root(value: str) -> str:
    root = value.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


def _json_response(response: httpx.Response, label: str) -> dict[str, Any]:
    response.raise_for_status()
    try:
        value = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{label} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} returned a non-object JSON body")
    return value


def resolve_api_key(*, api_key: str, api_key_env: str) -> str:
    """Resolve an optional inbound key without requiring it in process args."""
    if not api_key_env:
        return api_key
    value = os.environ.get(api_key_env, "")
    if not value:
        raise RuntimeError(f"adapter API key environment variable is empty: {api_key_env}")
    return value


def run_smoke(
    *,
    base_url: str,
    api_key: str = "",
    expected_model: str | None = None,
    allow_proxy_call: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Verify health, model discovery, and one complete chat response."""
    root = _service_root(base_url)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    owns_client = client is None
    active_client = client or httpx.Client(timeout=10)
    try:
        health = _json_response(active_client.get(f"{root}/health"), "health")
        if health.get("status") != "ok":
            raise RuntimeError("adapter health did not report status=ok")
        models_response = active_client.get(f"{root}/v1/models", headers=headers)
        models = _json_response(models_response, "models")
        mode = health.get("mode") or models_response.headers.get(
            "x-antisycophancy-adapter-mode"
        )
        if mode != "reference_response" and not allow_proxy_call:
            detail = "adapter is in proxy mode" if mode == "proxy" else "cannot prove reference mode"
            raise RuntimeError(
                f"{detail}; re-run with --allow-proxy-call to permit one upstream call"
            )
        rows = models.get("data")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("models response did not contain any model ids")
        model_ids = [row.get("id") for row in rows if isinstance(row, dict)]
        model = expected_model or str(model_ids[0] or "")
        if not model or model not in model_ids:
            raise RuntimeError(f"expected model is not advertised: {model}")

        chat = _json_response(
            active_client.post(
                f"{root}/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Adapter contract smoke."}],
                    "max_tokens": 32,
                },
            ),
            "chat completions",
        )
        choices = chat.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("chat completion did not contain assistant text")
        return {
            "status": "pass",
            "mode": mode,
            "model": model,
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "assistant_text_chars": len(content),
        }
    finally:
        if owns_client:
            active_client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog=(
            "Reference mode is local and free. In proxy mode, the chat probe is blocked unless "
            "--allow-proxy-call is present because the upstream request may be paid. "
            "Full setup and customization guide: adapter/README.md"
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:9999",
        help="Adapter service root or /v1 base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="",
        help="Read the inbound bearer key from this environment variable.",
    )
    parser.add_argument("--expected-model", help="Require this model id in /v1/models.")
    parser.add_argument(
        "--allow-proxy-call",
        action="store_true",
        help="Allow the smoke to make one potentially paid upstream model call.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        api_key = resolve_api_key(api_key="", api_key_env=args.api_key_env)
        result = run_smoke(
            base_url=args.base_url,
            api_key=api_key,
            expected_model=args.expected_model,
            allow_proxy_call=args.allow_proxy_call,
        )
    except (httpx.HTTPError, RuntimeError) as exc:
        print(f"adapter smoke failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
