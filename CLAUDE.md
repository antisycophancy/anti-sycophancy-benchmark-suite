# Claude Code Instructions

This repository contains the Anti-Sycophancy Benchmark Suite. Read
`AGENTS.md` before changing code or operating a benchmark.

## When the user wants to run the benchmark

Load `skills/antisycophancy/SKILL.md`. Use its onboarding guide for a new user
and `docs/AGENT_RUNBOOK.md` for the complete repository workflow. The user can
invoke the guide as `/antisycophancy`, but ordinary requests such as "help me
test Claude directly" should work without knowing the command name.

Begin continued work with this prompt-free coordination check.

```bash
./venv/bin/python -m suite_tools.companion resume --json
```

A missing workflow is normal for a new user. When a workflow exists, re-read
its `RUN_CONTRACT.json`, `RUN_STATUS.json`, and `RUN_EVENTS.jsonl`. The ledgers
are authoritative. Chat history and companion state are not.

## Default operating posture

- Explain SUS, AITA, and Epistemic in plain language before asking the user to
  choose.
- Recommend one model, one scenario or item, and one run for the first smoke.
- Prefer OpenRouter for the simplest multi-provider setup, but explain that it
  is a gateway. Support roles and judges may use different routes from the
  model under test.
- Never ask for a key in chat. Use the ignored repository-root `.env` file and
  check presence without printing the value.
- Treat preparation and scheduler dry-run as offline work.
- Treat external data-pack download, catalog lookup, exact-condition preflight,
  generation, and scoring as distinct external boundaries.
- Show the exact models, endpoints, support roles, expected calls, estimate,
  data route, and command before requesting paid approval.
- Keep generation and scoring as separate approvals unless the user explicitly
  approves automatic scoring for the exact prepared contract.
- Start the dashboard in its own terminal before generation unless the user
  declines it, then give them `http://127.0.0.1:8765`.
- Never edit a prepared contract or completed artifact. Resume only work that
  the immutable contract still identifies as owed.
- Require `bench verify` and `hygiene_gate` before scoring.
- Do not package unresolved, non-publishable, drifting, or privacy-unsafe
  evidence.

## Current command truth

Do not rely on flags remembered from another checkout. Read current help.

```bash
./venv/bin/python -m suite_tools.prepare_run --help
./venv/bin/python -m suite_tools.preflight_conditions --help
./venv/bin/python -m suite_tools.scheduler --help
./venv/bin/python -m suite_tools.bench --help
```

Use `suite_models.yaml`, `suite_tools.prepare_run`, and
`suite_tools.scheduler` for current comparable runs. Use the canonical
`adapter/` only when a backend does not already accept OpenAI Chat Completions.

## Before interpreting a result

Read the main README section `Scope and limitations`. Carry those limits into
any summary. A small run is evidence about one frozen condition, not a general
model ranking or safety certification.
