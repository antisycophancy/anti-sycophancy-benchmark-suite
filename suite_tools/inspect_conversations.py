"""Print saved benchmark conversations as readable Markdown for manual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.rglob("*.json")
        if "_scores" not in p.name and p.name not in {"FINAL_RESULTS.json", "manifest.json"}
    )


def _load(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _iter_records(path: Path) -> Iterable[tuple[Path, dict]]:
    for file in _json_files(path):
        try:
            data = _load(file)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    yield file, row
        elif isinstance(data, dict) and isinstance(data.get("results"), list):
            for row in data["results"]:
                if isinstance(row, dict):
                    yield file, row
        elif isinstance(data, dict):
            yield file, data


def _turns(record: dict) -> list[tuple[str, str]]:
    if isinstance(record.get("conversation"), list):
        return [
            (str(msg.get("role", "")).upper(), str(msg.get("content", "")))
            for msg in record["conversation"]
            if isinstance(msg, dict)
        ]

    if isinstance(record.get("turns"), list):
        turns = []
        for turn in record["turns"]:
            if not isinstance(turn, dict):
                continue
            turns.append(("USER", str(turn.get("user_message", ""))))
            turns.append(("ASSISTANT", str(turn.get("model_response", ""))))
        return turns

    return []


def _title(record: dict, source: Path) -> str:
    bits = [
        record.get("label") or record.get("model") or record.get("model_id"),
        record.get("scenario_name") or record.get("scenario") or record.get("test_type"),
        record.get("side"),
        f"item {record.get('item_idx')}" if record.get("item_idx") is not None else None,
    ]
    label = " | ".join(str(bit) for bit in bits if bit)
    return label or source.name


def render_markdown(path: Path, *, limit: int | None = None) -> str:
    lines = ["# Conversation Review", ""]
    count = 0
    for source, record in _iter_records(path):
        turns = _turns(record)
        if not turns:
            continue
        count += 1
        lines.extend([
            f"## {count}. {_title(record, source)}",
            "",
            f"Source: `{source}`",
            "",
        ])
        score = record.get("score")
        if isinstance(score, dict):
            lines.append(f"Score: `{json.dumps(score, sort_keys=True)}`")
            lines.append("")
        for role, content in turns:
            lines.append(f"**{role}:**")
            lines.append("")
            lines.append(content.strip() or "[empty]")
            lines.append("")
        if limit is not None and count >= limit:
            break
    if count == 0:
        lines.append("_No conversations found._")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render saved benchmark conversations as Markdown.")
    parser.add_argument("path", help="Conversation JSON file or directory.")
    parser.add_argument("--limit", type=int, help="Maximum number of conversations to print.")
    parser.add_argument("--output", "-o", help="Optional Markdown output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = render_markdown(Path(args.path), limit=args.limit)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
