# Benchmark Agent Instructions

This repo contains public benchmark modules, shared suite tooling, and local
operator utilities for anti-sycophancy evaluation.

## First-Run Guidance

- For requests to understand, install, configure, connect a model to, or run
  the suite, load `skills/antisycophancy/SKILL.md`. Start with its
  `references/getting-started.md` when the user is new or speaking in plain
  language. Use `docs/AGENT_RUNBOOK.md` for the complete clone-to-result flow.
- Claude Code users can invoke the installed skill as
  `/antisycophancy`; Codex users invoke it as
  `$antisycophancy`. Both should also recognize ordinary requests
  such as "help me test Claude directly" or "connect my model server."
- Guide one stage at a time. Explain the three modules, provider choices, key
  location, data route, and paid boundaries before presenting advanced
  operational detail.
- For a new operator, recommend one model, one scenario or item, and one run.
  Show the prepared contract, exact endpoint and support-role targets, pricing
  estimate, and next command before requesting any external call.
- Treat external data-pack download, catalog lookup, exact-condition preflight,
  generation, and scoring as separate boundaries. Keep generation and scoring
  under separate approvals unless the user explicitly approves automatic
  scoring for that exact frozen contract.
- Start `suite_tools.live_dashboard` in its own terminal before generation
  unless the user declines it, then give them `http://127.0.0.1:8765`. The
  dashboard is an operator view. Run ledgers remain authoritative.
- At the beginning of continued benchmark work, run
  `./venv/bin/python -m suite_tools.companion resume --json`. A missing active
  workflow is normal for a new user. Treat its receipt as coordination context
  only and re-read the attached run ledgers before acting.

## Working Rules

- Keep public benchmark code provider-neutral whenever possible.
- Keep private prompts, service IDs, backend routing, database credentials,
  traces, unpublished question banks, and ad hoc run artifacts out of tracked
  source.
- Use ignored `internal/`, `private_profiles/`, `private_question_bank/`, and
  `results/` paths for local/private operations.
- Treat private served systems as normal OpenAI-compatible endpoints from the
  benchmark's perspective. Public artifacts may record model ids, condition
  hashes, response text, scoring metadata, and public labels; they should not
  expose private routing or prompt internals.
- Capacity scaling is operational only. It must not alter benchmark questions,
  prompts, request payloads, model ids, judges, scoring code, or promotion
  gates.
- Prefer `suite_models.yaml`, `suite_tools.prepare_run`, and
  `suite_tools.scheduler` for current runs.
- Before any wide/paid run, read RUNBOOK section 0.1 (capacity is min(policy, `.env`)
  AND the scheduler must request the ceiling via `--max-active-calls N`) and
  section 0.6 (reasoning-effort endpoints, `preflight_conditions` before spend,
  uniform output-token cap, resume/`clear-control`, and the quota vs rate-limit
  vs content-block failure taxonomy).
- On resume, reuse only artifacts whose saved `condition_id` and
  `condition_hash` match the same rendered condition in the frozen contract.
  Run `bench verify` before scoring and require artifact identity conformance;
  never repair a completed source run in place.
- Preserve unrelated local changes and avoid broad cleanup from the parent
  workspace.
- When reporting or interpreting scores, carry the caveats from
  `README.md` §"Scope and limitations" (one SUS scenario, unvalidated judges,
  n=20 Wilson intervals 33.5-40.1pp wide, unpinned temperature, refusal scores
  well). `docs/HARDENING_BACKLOG.md` is the maintained list of known
  methodology gaps behind them.

## Registry, Review, and Publication

`suite_tools.bench` is the unified registry and packaging CLI for post-run
management. All commands use `./venv/bin/python -m suite_tools.bench <verb>` and accept
`--json`.  Use `--root <dir>` on scanning verbs to add non-default search roots.

| Verb | Purpose |
|------|---------|
| `runs` | Scan for run directories in default roots. |
| `experiments` | Find every `EXPERIMENT.json` in the same roots. |
| `status <run_dir>` | Show owed units for one run. |
| `diagnose <run_dir>` | Explain private provider-call failures and ambiguous attempts. |
| `blockers` | List runs whose latest attempt halted. |
| `adopt <exp_dir> <run_dir> --role ROLE` | Adopt a completed run into an experiment. |
| `supersede <exp_dir> <member> --by <member> --reason REASON` | Mark a member as superseded. |
| `verify [<run_dir>]` | Recompute hashes, check comparability, or audit a bundle (`--bundle <dir>`). |
| `package <exp_dir> --out <dir>` | Emit a self-contained experiment bundle (gated). |
| `review` | Triage the unresolved evidence queue before packaging. |

### Evidence-review triage

After all modules reach `validity=score_ready`, use `bench review` to
disposition any unresolved provider failures or model-signal blocks before
packaging:

```bash
# List open items; gate_blocking=true rows block bench package
./venv/bin/python -m suite_tools.bench review --json

# Disposition one fact
./venv/bin/python -m suite_tools.bench review \
  --run results/prepared/RUN_ID/sus \
  --event-ref blocks-id:<uuid> \
  --by <reviewer-id> --reason "reason text" \
  --disposition safety_declination   # or: retry | instrument_defect | needs_escalation

# Emit the bundle once no gate_blocking rows remain
./venv/bin/python -m suite_tools.bench package results/experiments/exp-id --out results/bundles --json
```

`bench package` is fail-closed. It refuses when any member's `RUN_STATUS.status`
is not `"completed"`, or when any unreviewed facts, active `needs_escalation`
reviews, or non-publishable unit states remain.  A fingerprint-drift abort
(mid-bundle file change) requires settling the in-progress attempt and
re-running. See RUNBOOK sections 0.7 and 0.8 for the full reference.

## Checks

```bash
./venv/bin/python -m suite_tools.offline_gate
./venv/bin/python -m pytest -q tests unified_profile/tests
```
