"""Local-only diagnostics for the reference adapter's upstream boundary."""

from __future__ import annotations

import fcntl
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "adapter-diagnostic-v1"


def _sanitize(value: object) -> str:
    text = str(value)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]{8,}", "Bearer <redacted>", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]{16,}", "sk-<redacted>", text)
    text = re.sub(r"AIza[0-9A-Za-z_-]{30,}", "AIza<redacted>", text)
    return text[:1000]


def _rotate(path: Path, max_bytes: int) -> None:
    try:
        if path.stat().st_size < max_bytes:
            return
    except OSError:
        return
    for index in range(3, 1, -1):
        older = path.with_name(f"{path.name}.{index - 1}")
        newer = path.with_name(f"{path.name}.{index}")
        if older.exists():
            os.replace(older, newer)
    os.replace(path, path.with_name(f"{path.name}.1"))


def record_adapter_error(
    path: Path | None,
    *,
    request_id: str,
    model: str,
    error: Any,
    include_detail: bool = False,
    max_bytes: int = 16 * 1024 * 1024,
) -> None:
    """Append an adapter-authored error claim without changing HTTP output."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    context = getattr(error, "context", None)
    if not isinstance(context, dict):
        context = {}
    safe_context = {
        key: context[key]
        for key in (
            "upstream_status",
            "raw_response_sha256",
            "response_shape",
            "finish_reason",
            "native_finish_reason",
        )
        if context.get(key) is not None
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "claim_source": "adapter_claim",
        "request_id": request_id,
        "model": str(model),
        "status_code": int(getattr(error, "status_code", 500)),
        "code": str(getattr(error, "code", type(error).__name__)),
        "message": _sanitize(error),
        "context": safe_context,
    }
    detail = getattr(error, "detail", None)
    if include_detail and detail is not None:
        payload["local_detail"] = _sanitize(detail)

    lock_path = path.with_name(f"{path.name}.lock")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _rotate(path, max(max_bytes, 1024))
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.chmod(path, 0o600)
            os.write(fd, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
