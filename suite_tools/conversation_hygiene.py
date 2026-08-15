"""Shared conversation hygiene checks for benchmark artifacts.

This module is intentionally benchmark-agnostic. SUS stores OpenAI-style
``conversation`` lists, while AITA and Epistemic store turn rows with
``user_message`` and ``model_response``. The scanner normalizes those shapes
and applies one failure grammar before artifacts are scored or published.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from suite_tools.suite_registry import module_key_for_record

Severity = Literal["blocking", "review"]

ERROR_STATUS_RE = re.compile(
    r"\b(?:http(?:/[\d.]+)?|status(?:\s+code)?|code|error|returned|response)\s*[:#-]?\s*"
    r"(?P<code>400|401|403|404|408|409|429|500|502|503|504)\b|"
    r"\b(?P<prefix_code>400|401|403|404|408|409|429|500|502|503|504)\s+"
    r"(?:error|status|response)\b",
    re.IGNORECASE,
)
BRACKETED_ERROR_RE = re.compile(r"^\[(?:ERROR|TIMEOUT/ERROR)(?:\s+\d+)?(?::|\b)", re.IGNORECASE)
EMPTY_RESPONSE_SENTINELS = {
    "[empty response from model]",
    "[empty response]",
    "[no response]",
    "(empty response)",
}
PROVIDER_ERROR_RE = re.compile(
    r"backend timeout|backend returned|readtimeout|timed out|timeout/error|"
    r"bad gateway|cloudflare|jwt validation|not configured|"
    r"unauthorized (?:api|request|token|key|client|endpoint|model|provider|error)|"
    r"authorization failed|authentication failed|invalid api key|"
    r"not a valid model id|invalid model id|\bkeyerror\b|\btraceback\b|"
    r"\b(?:runtime|uncaught|internal|provider|api|server|python) exception\b|"
    r"\bexception (?:occurred|while|during)\b|"
    r"\bexception in (?:adapter|api|backend|client|endpoint|model|provider|python|request|response|runtime|server|thread)\b|"
    r"adapter rejected|adapter_incomplete|backend_non_200",
    re.IGNORECASE,
)
ERROR_WORD_RE = re.compile(
    r"\berror\b|timeout|bad gateway|invalid|unauthorized|jwt|not configured|exception|adapter rejected",
    re.IGNORECASE,
)
HTML_ERROR_RE = re.compile(r"<html|<!doctype html|<title>.*(?:error|bad gateway)", re.IGNORECASE | re.DOTALL)
RESPONSE_WRAPPER_KEYS = {"response", "output", "text", "response_text", "message", "content"}
ERROR_JSON_KEYS = {"error", "detail", "errors"}
METADATA_KEYS = {
    "analysis",
    "clinical_note",
    "debug",
    "finish_reason",
    "model_stance",
    "reasoning",
    "risk_flag",
    "thought",
    "trace",
    "user_claim",
}
CLINICAL_NOTE_TAG_RE = re.compile(r"</?\s*clinical[_ -]?note\b", re.IGNORECASE)
CLINICAL_NOTE_KEY_RE = re.compile(r"(?P<quote>[\"'])?clinical_note(?P=quote)?\s*[:=]", re.IGNORECASE)
CLINICAL_NOTE_TOKEN_RE = re.compile(r"\bclinical[_ -]?note\b", re.IGNORECASE)
SIDECAR_SCHEMA_KEY_RE = re.compile(
    r"(?P<quote>[\"'])?(user_claim|model_stance|advisor_position|framing_change|risk_flag)(?P=quote)?\s*[:=]",
    re.IGNORECASE,
)
SIDECAR_SCHEMA_TOKEN_RE = re.compile(
    r"\b(user_claim|model_stance|advisor_position|framing_change|risk_flag)\b",
    re.IGNORECASE,
)
USER_KEYS = ("user_message", "user", "prompt", "question")
ASSISTANT_KEYS = ("model_response", "assistant_response", "assistant", "response", "answer", "completion")
SKIP_JSON_NAMES = {
    "FINAL_RESULTS.json",
    "RUN_STATUS.json",
    "RUN_CONTROL.json",
    "RUN_CONTRACT.json",
    "manifest.json",
    "dashboard_data.json",
}

HUMAN_REVIEW_META_PROMPT = """You are auditing benchmark conversation artifacts before they enter a scored dataset.

Return JSON only:
{
  "verdict": "clean" | "review" | "block",
  "issues": [
    {
      "severity": "blocking" | "review",
      "turn": <number or null>,
      "role": "user" | "assistant" | "unknown",
      "code": "<short_code>",
      "reason": "<why this is not a normal benchmark chat message>"
    }
  ]
}

Use "block" if any model/assistant response is empty, a bracketed runtime
sentinel, timeout, exception, HTTP/provider error, raw error JSON, HTML error
page, malformed JSON-like text, unexpected role, missing content, incomplete
conversation, or broken turn ordering. Use "review" if the answer is
semantically usable but not a clean chat transcript, including JSON response
wrappers, clinical/debug notes, thought/analysis fields, risk flags, or other
metadata leakage. Use "clean" only when every model response is normal chat
text for the benchmark.
"""


@dataclass(frozen=True)
class NormalizedTurn:
    """A model/user message normalized from one benchmark artifact shape."""

    role: str
    content: Any
    location: str
    enforce_order: bool = True


@dataclass(frozen=True)
class HygieneIssue:
    """One transcript hygiene issue."""

    severity: Severity
    code: str
    message: str
    record_index: int
    source: str
    module: str
    model: str
    scenario: str
    run_number: Any
    location: str
    role: str
    excerpt: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable issue row."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "record_index": self.record_index,
            "source": self.source,
            "module": self.module,
            "model": self.model,
            "scenario": self.scenario,
            "run_number": self.run_number,
            "location": self.location,
            "role": self.role,
            "excerpt": self.excerpt,
        }


def excerpt(value: Any, limit: int = 280) -> str:
    """Return a compact one-line excerpt."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def _module_for(record: dict[str, Any], source: str) -> str:
    return module_key_for_record(record, source)


def _record_meta(record: dict[str, Any], record_index: int, source: str) -> dict[str, Any]:
    scenario = record.get("scenario") or record.get("scenario_name") or record.get("test_type") or ""
    if isinstance(scenario, dict):
        scenario = scenario.get("id") or scenario.get("title") or json.dumps(scenario, sort_keys=True)
    return {
        "record_index": record_index,
        "source": source,
        "module": _module_for(record, source),
        "model": str(record.get("model") or record.get("model_id") or record.get("label") or ""),
        "scenario": str(scenario),
        "run_number": record.get("run_number", record.get("runNumber", record.get("item_idx", ""))),
    }


def _issue(
    record: dict[str, Any],
    record_index: int,
    source: str,
    *,
    severity: Severity,
    code: str,
    message: str,
    location: str,
    role: str = "",
    content: Any = "",
) -> HygieneIssue:
    meta = _record_meta(record, record_index, source)
    return HygieneIssue(
        severity=severity,
        code=code,
        message=message,
        location=location,
        role=role,
        excerpt=excerpt(content),
        **meta,
    )


def normalized_turns(record: dict[str, Any]) -> list[NormalizedTurn]:
    """Extract user/assistant messages from known benchmark artifact shapes."""
    for key in ("conversation", "messages"):
        if isinstance(record.get(key), list):
            turns: list[NormalizedTurn] = []
            for idx, msg in enumerate(record[key], start=1):
                if isinstance(msg, dict):
                    turns.append(
                        NormalizedTurn(
                            role=str(msg.get("role") or ""),
                            content=msg.get("content"),
                            location=f"{key}[{idx}]",
                        )
                    )
                else:
                    turns.append(
                        NormalizedTurn(
                            role="",
                            content=msg,
                            location=f"{key}[{idx}]",
                        )
                    )
            return turns

    if isinstance(record.get("turns"), list):
        turns = []
        for idx, turn in enumerate(record["turns"], start=1):
            if not isinstance(turn, dict):
                turns.append(NormalizedTurn(role="", content=turn, location=f"turns[{idx}]"))
                continue
            user_found = False
            for key in USER_KEYS:
                if key in turn:
                    turns.append(
                        NormalizedTurn("user", turn.get(key), f"turns[{idx}].{key}")
                    )
                    user_found = True
                    break
            assistant_found = False
            assistant_turn: NormalizedTurn | None = None
            for key in ASSISTANT_KEYS:
                if key in turn:
                    assistant_turn = NormalizedTurn("assistant", turn.get(key), f"turns[{idx}].{key}")
                    turns.append(assistant_turn)
                    assistant_found = True
                    break
            if not user_found and not assistant_found:
                turns.append(
                    NormalizedTurn(
                        role="",
                        content=turn,
                        location=f"turns[{idx}]",
                        enforce_order=False,
                    )
                )
            elif not user_found:
                missing_user = NormalizedTurn(
                    role="user",
                    content="",
                    location=f"turns[{idx}].user_message",
                )
                if assistant_turn is not None:
                    turns.insert(len(turns) - 1, missing_user)
                else:
                    turns.append(missing_user)
            elif not assistant_found:
                turns.append(
                    NormalizedTurn(
                        role="assistant",
                        content="",
                        location=f"turns[{idx}].model_response",
                    )
                )
        return turns

    turns = []
    for key in USER_KEYS:
        if key in record:
            turns.append(NormalizedTurn("user", record.get(key), key, enforce_order=False))
            break
    for key in ASSISTANT_KEYS:
        if key in record:
            turns.append(NormalizedTurn("assistant", record.get(key), key, enforce_order=False))
            break
    return turns


def canonical_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return the model-readable transcript surface as role/content messages.

    Benchmarks may persist richer turn rows for resume/scoring metadata, but
    judges, viewers, and hygiene gates should consume this canonical chat-like
    shape whenever possible.
    """
    messages: list[dict[str, str]] = []
    for turn in normalized_turns(record):
        if turn.role not in {"user", "assistant"}:
            continue
        content = turn.content if isinstance(turn.content, str) else ""
        messages.append({"role": turn.role, "content": content})
    return messages


def render_chat_transcript(record: dict[str, Any]) -> str:
    """Render a canonical role-labeled transcript for judge/meta-review prompts."""
    lines: list[str] = []
    for index, message in enumerate(canonical_messages(record), start=1):
        role = message["role"].upper()
        lines.append(f"{index}. {role}\n{message['content'].strip()}")
    return "\n\n".join(lines)


def inspect_text(
    content: Any,
    *,
    role: str,
    location: str,
    record: dict[str, Any],
    record_index: int,
    source: str,
) -> list[HygieneIssue]:
    """Inspect one normalized message text field."""
    issues: list[HygieneIssue] = []
    if content is None:
        return [
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="schema.missing_content",
                message="Message content is missing.",
                location=location,
                role=role,
                content="",
            )
        ]
    if not isinstance(content, str):
        return [
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="schema.non_string_content",
                message="Message content is not a string.",
                location=location,
                role=role,
                content=repr(content),
            )
        ]

    stripped = content.strip()
    if not stripped:
        return [
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="schema.empty_content",
                message="Message content is empty.",
                location=location,
                role=role,
                content=content,
            )
        ]

    if role != "assistant":
        return []

    lowered = stripped.lower()
    if role == "assistant" and lowered in EMPTY_RESPONSE_SENTINELS:
        issues.append(
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="assistant.empty_response_sentinel",
                message="Assistant/model turn is an empty-response sentinel, not model behavior.",
                location=location,
                role=role,
                content=content,
            )
        )

    if BRACKETED_ERROR_RE.search(stripped):
        issues.append(
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="assistant.bracketed_error_sentinel",
                message="Assistant/model turn is a bracketed runtime/provider error sentinel.",
                location=location,
                role=role,
                content=content,
            )
        )

    status_codes = [
        match.group("code") or match.group("prefix_code")
        for match in ERROR_STATUS_RE.finditer(stripped)
    ]
    if status_codes and ERROR_WORD_RE.search(stripped):
        issues.append(
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="assistant.status_error_text",
                message=f"Assistant/model turn contains HTTP/provider error status code(s): {', '.join(sorted(set(status_codes)))}.",
                location=location,
                role=role,
                content=content,
            )
        )

    if PROVIDER_ERROR_RE.search(stripped):
        issues.append(
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="assistant.provider_error_text",
                message="Assistant/model turn contains provider/runtime failure text.",
                location=location,
                role=role,
                content=content,
            )
        )

    if HTML_ERROR_RE.search(stripped):
        issues.append(
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="assistant.html_error_page",
                message="Assistant/model turn contains an HTML error page.",
                location=location,
                role=role,
                content=content,
            )
        )

    if stripped.startswith("{"):
        issues.extend(
            _inspect_json_like(
                stripped,
                role=role,
                location=location,
                record=record,
                record_index=record_index,
                source=source,
            )
        )

    if _looks_like_visible_sidecar(stripped):
        issues.append(
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="assistant.visible_sidecar_leak",
                message="Assistant/model turn exposes backend-only clinical-note sidecar fields.",
                location=location,
                role=role,
                content=content,
            )
        )

    return _dedupe_issues(issues)


def _looks_like_visible_sidecar(text: str) -> bool:
    if (
        CLINICAL_NOTE_TAG_RE.search(text)
        or CLINICAL_NOTE_KEY_RE.search(text)
        or CLINICAL_NOTE_TOKEN_RE.search(text)
    ):
        return True
    return (
        len(SIDECAR_SCHEMA_KEY_RE.findall(text)) >= 2
        or len(SIDECAR_SCHEMA_TOKEN_RE.findall(text)) >= 2
    )


def _inspect_json_like(
    text: str,
    *,
    role: str,
    location: str,
    record: dict[str, Any],
    record_index: int,
    source: str,
) -> list[HygieneIssue]:
    issues: list[HygieneIssue] = []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="assistant.malformed_json_like",
                message="Assistant/model turn starts like JSON but is not valid JSON.",
                location=location,
                role=role,
                content=text,
            )
        ]

    if not isinstance(parsed, dict):
        return [
            _issue(
                record,
                record_index,
                source,
                severity="review",
                code="assistant.json_non_object",
                message="Assistant/model turn is JSON rather than natural chat text.",
                location=location,
                role=role,
                content=text,
            )
        ]

    keys = {str(key).lower() for key in parsed}
    error_keys = keys & ERROR_JSON_KEYS
    if error_keys:
        issues.append(
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="assistant.raw_error_json",
                message=f"Assistant/model turn is raw error JSON with key(s): {', '.join(sorted(error_keys))}.",
                location=location,
                role=role,
                content=text,
            )
        )

    wrapper_keys = keys & RESPONSE_WRAPPER_KEYS
    if wrapper_keys:
        issues.append(
            _issue(
                record,
                record_index,
                source,
                severity="review",
                code="assistant.structured_response_wrapper",
                message=f"Assistant/model turn is a JSON response wrapper with key(s): {', '.join(sorted(wrapper_keys))}.",
                location=location,
                role=role,
                content=text,
            )
        )

    metadata_keys = keys & METADATA_KEYS
    if metadata_keys:
        issues.append(
            _issue(
                record,
                record_index,
                source,
                severity="review",
                code="assistant.metadata_or_thought_leak",
                message=f"Assistant/model turn exposes metadata/debug fields: {', '.join(sorted(metadata_keys))}.",
                location=location,
                role=role,
                content=text,
            )
        )

    if not issues:
        issues.append(
            _issue(
                record,
                record_index,
                source,
                severity="review",
                code="assistant.json_like_text",
                message="Assistant/model turn is JSON-shaped rather than natural chat text.",
                location=location,
                role=role,
                content=text,
            )
        )

    return issues


def scan_record(record: dict[str, Any], *, record_index: int = 0, source: str = "") -> list[HygieneIssue]:
    """Scan one benchmark result record."""
    issues: list[HygieneIssue] = []
    if not isinstance(record, dict):
        return [
            HygieneIssue(
                severity="blocking",
                code="schema.record_not_object",
                message="Result record is not an object.",
                record_index=record_index,
                source=source,
                module="generic",
                model="",
                scenario="",
                run_number="",
                location="record",
                role="",
                excerpt=excerpt(repr(record)),
            )
        ]

    issues.extend(_completion_issues(record, record_index=record_index, source=source))
    turns = normalized_turns(record)
    if not turns:
        return _dedupe_issues(issues)

    expected_role = "user"
    for turn in turns:
        role = str(turn.role or "")
        if role not in {"user", "assistant"}:
            issues.append(
                _issue(
                    record,
                    record_index,
                    source,
                    severity="blocking",
                    code="schema.unexpected_role",
                    message="Conversation role is not user or assistant.",
                    location=turn.location,
                    role=role,
                    content=turn.content,
                )
            )
        elif turn.enforce_order and role != expected_role:
            issues.append(
                _issue(
                    record,
                    record_index,
                    source,
                    severity="blocking",
                    code="schema.broken_turn_order",
                    message=f"Expected {expected_role} turn but found {role}.",
                    location=turn.location,
                    role=role,
                    content=turn.content,
                )
            )

        issues.extend(
            inspect_text(
                turn.content,
                role=role,
                location=turn.location,
                record=record,
                record_index=record_index,
                source=source,
            )
        )
        if turn.enforce_order:
            expected_role = "assistant" if role == "user" else "user"

    issues.extend(_scan_phase_text(record, record_index=record_index, source=source))
    return _dedupe_issues(issues)


def _completion_issues(record: dict[str, Any], *, record_index: int, source: str) -> list[HygieneIssue]:
    issues: list[HygieneIssue] = []
    if record.get("completed") is False:
        issues.append(
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="schema.incomplete_conversation",
                message="Conversation artifact is marked incomplete.",
                location="completed",
                content=record.get("failure_reason") or "completed=false",
            )
        )
    planned = record.get("planned_num_turns") or record.get("num_turns")
    actual = record.get("actual_num_turns")
    if isinstance(planned, int) and isinstance(actual, int) and actual < planned:
        issues.append(
            _issue(
                record,
                record_index,
                source,
                severity="blocking",
                code="schema.short_conversation",
                message=f"Conversation has {actual} turns but expected {planned}.",
                location="actual_num_turns",
                content=f"{actual}/{planned}",
            )
        )
    return issues


def _scan_phase_text(record: dict[str, Any], *, record_index: int, source: str) -> list[HygieneIssue]:
    phases = record.get("phases")
    if not isinstance(phases, dict):
        return []

    issues: list[HygieneIssue] = []
    seen_text = {
        str(turn.content or "").strip()
        for turn in normalized_turns(record)
    }
    for phase_key, phase in phases.items():
        if not isinstance(phase, dict):
            continue
        for field, role in (("response", "assistant"), ("prompt", "user")):
            if field not in phase:
                continue
            content = phase.get(field)
            if not isinstance(content, str) or not content.strip():
                continue
            if content.strip() in seen_text:
                continue
            issues.extend(
                inspect_text(
                    content,
                    role=role,
                    location=f"phases.{phase_key}.{field}",
                    record=record,
                    record_index=record_index,
                    source=source,
                )
            )
    return issues


def _json_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        candidates: list[Path]
        if path.is_file() and path.suffix == ".json":
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(
                file
                for file in path.rglob("*.json")
                if file.name not in SKIP_JSON_NAMES
            )
        else:
            candidates = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(candidate)
    return files


def _load_json(path: Path) -> Any | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def iter_records(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    """Yield candidate benchmark records from files or directories."""
    for path in _json_files(paths):
        data = _load_json(path)
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    yield path, row
            continue
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("results"), list):
            for row in data["results"]:
                if isinstance(row, dict):
                    yield path, row
            continue
        if isinstance(data.get("cases"), list):
            root_bits = {k: v for k, v in data.items() if k != "cases"}
            for case in data["cases"]:
                if isinstance(case, dict):
                    yield path, {**root_bits, **case}
            continue
        if normalized_turns(data) or data.get("completed") is False:
            yield path, data


def scan_results(results: Iterable[dict[str, Any]], *, source: str = "") -> list[HygieneIssue]:
    """Scan many already-loaded benchmark records."""
    issues: list[HygieneIssue] = []
    for record_index, record in enumerate(results, start=1):
        issues.extend(scan_record(record, record_index=record_index, source=source))
    return issues


def scan_paths(paths: Iterable[Path]) -> list[HygieneIssue]:
    """Scan all candidate records under files/directories."""
    issues: list[HygieneIssue] = []
    per_source_index: dict[Path, int] = {}
    for source, record in iter_records(paths):
        per_source_index[source] = per_source_index.get(source, 0) + 1
        issues.extend(
            scan_record(
                record,
                record_index=per_source_index[source],
                source=str(source),
            )
        )
    return issues


def _dedupe_issues(issues: Iterable[HygieneIssue]) -> list[HygieneIssue]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[HygieneIssue] = []
    for issue in issues:
        key = (
            issue.severity,
            issue.code,
            issue.record_index,
            issue.source,
            issue.location,
            issue.role,
            issue.excerpt,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return unique


def summarize_issues(issues: Iterable[HygieneIssue]) -> dict[str, Any]:
    """Summarize issues for console and JSON reports."""
    issue_list = list(issues)
    blocking = [issue for issue in issue_list if issue.severity == "blocking"]
    review = [issue for issue in issue_list if issue.severity == "review"]
    records_with_blocking = {(issue.source, issue.record_index) for issue in blocking}
    records_with_review = {(issue.source, issue.record_index) for issue in review}
    by_code: dict[str, int] = {}
    by_module: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for issue in issue_list:
        by_code[issue.code] = by_code.get(issue.code, 0) + 1
        by_module[issue.module] = by_module.get(issue.module, 0) + 1
        by_source[issue.source] = by_source.get(issue.source, 0) + 1
    return {
        "issues": len(issue_list),
        "blocking_issues": len(blocking),
        "review_issues": len(review),
        "records_with_blocking": len(records_with_blocking),
        "records_with_review": len(records_with_review),
        "by_code": dict(sorted(by_code.items())),
        "by_module": dict(sorted(by_module.items())),
        "by_source": dict(sorted(by_source.items())),
    }


def blocking_issue_summaries(
    record: dict[str, Any],
    *,
    source: str | Path = "",
    record_index: int = 0,
) -> list[str]:
    """Return short operator-facing lines for blocking hygiene issues."""
    source_text = str(source)
    lines: list[str] = []
    for issue in scan_record(record, source=source_text, record_index=record_index):
        if issue.severity != "blocking":
            continue
        lines.append(
            f"{issue.source}: {issue.location}: {issue.code}: "
            f"{issue.message} ({issue.excerpt})"
        )
    return lines


def write_json_report(path: Path, issues: Iterable[HygieneIssue]) -> None:
    """Write a machine-readable hygiene report."""
    issue_list = list(issues)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "summary": summarize_issues(issue_list),
                "issues": [issue.as_dict() for issue in issue_list],
                "human_review_meta_prompt": HUMAN_REVIEW_META_PROMPT,
            },
            indent=2,
        )
        + "\n"
    )


def write_csv_report(path: Path, issues: Iterable[HygieneIssue]) -> None:
    """Write issue rows as CSV."""
    issue_list = list(issues)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "severity",
                "code",
                "message",
                "record_index",
                "source",
                "module",
                "model",
                "scenario",
                "run_number",
                "location",
                "role",
                "excerpt",
            ],
        )
        writer.writeheader()
        for issue in issue_list:
            writer.writerow(issue.as_dict())


def write_markdown_report(path: Path, issues: Iterable[HygieneIssue]) -> None:
    """Write a human-readable hygiene report."""
    issue_list = list(issues)
    summary = summarize_issues(issue_list)
    lines = [
        "# Benchmark Conversation Hygiene Report",
        "",
        "## Summary",
        "",
        f"- Issues: {summary['issues']}",
        f"- Blocking issues: {summary['blocking_issues']}",
        f"- Review issues: {summary['review_issues']}",
        f"- Records with blocking issues: {summary['records_with_blocking']}",
        f"- Records with review issues: {summary['records_with_review']}",
        "",
        "## Modules",
        "",
    ]
    lines.extend(f"- `{module}`: {count}" for module, count in summary["by_module"].items())
    lines.extend(["", "## Codes", ""])
    lines.extend(f"- `{code}`: {count}" for code, count in summary["by_code"].items())
    lines.extend(["", "## Issues", ""])
    for issue in issue_list:
        lines.extend(
            [
                f"### {issue.severity.upper()} / `{issue.code}`",
                "",
                f"- Source: `{issue.source}`",
                f"- Module: `{issue.module}`",
                f"- Record: {issue.record_index}; model: `{issue.model}`; scenario: `{issue.scenario}`; run/item: `{issue.run_number}`",
                f"- Location: `{issue.location}`; role: `{issue.role}`",
                f"- Reason: {issue.message}",
                f"- Excerpt: {issue.excerpt}",
                "",
            ]
        )
    lines.extend(
        [
            "## Optional LLM Review Prompt",
            "",
            "```text",
            HUMAN_REVIEW_META_PROMPT.strip(),
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
