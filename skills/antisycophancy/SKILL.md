---
name: antisycophancy
description: Use when someone wants to understand, set up, connect a model to, or run the Anti-Sycophancy Benchmark Suite, including first-time onboarding, provider and API-key choices, adapters, paid collections, recovery, review, and evidence packaging.
---

# Anti-Sycophancy Benchmark Guide

Operate SUS, AITA, and Epistemic runs without weakening their contracts or
turning partial output into official evidence.

Invocation arguments choose a workflow without creating separate skills:
`connect` onboards an endpoint, `run` prepares and executes an approved scope,
`resume` rehydrates an existing workflow, `review` triages evidence, and
`package` verifies and emits a gated evidence bundle. In Claude Code these are
`/antisycophancy <mode>`; in Codex use `$antisycophancy <mode>`.

## Welcome People First

When the user is new, uncertain, or asking in ordinary language, start in
onboarding mode. Read [`references/getting-started.md`](references/getting-started.md)
and act as a guide, not a command generator.

- Explain in a few sentences what the suite tests and what each module adds.
- Recommend the smallest sensible first run rather than asking the user to
  design a benchmark grid.
- Help choose among OpenRouter, a provider-direct API, an existing
  OpenAI-compatible endpoint, and the bundled adapter.
- Explain each technical term before using it. "Endpoint" means the web address
  that receives model requests; "model ID" means the name that address expects.
- Ask one or two short questions at a time. Prefer choices with a recommendation
  over a long intake form.
- Never ask the user to paste an API key into chat. Show where to place it in
  the ignored `.env` file, then check only whether the variable is present.
- Separate four consent boundaries: external data-pack download,
  connection preflight, benchmark generation, and transcript judging. Each can make a
  different external request or expose different data.
- Do not run a paid command merely because onboarding began. Show the next paid
  step and obtain approval unless the user already approved that exact scope.

Natural-language requests such as "help me test Claude directly," "connect my
model server," or "run the cheapest first test" should trigger this mode. The
user does not need to know module names, provider APIs, or CLI flags.

## Start With Repository Truth

1. Confirm the checkout and read its `AGENTS.md`.
2. Inspect `git status --short`. Preserve unrelated changes and private files.
3. Read `RUNBOOK.md` sections 0.1, 0.6, and 0.7 before a wide or paid run.
4. Read `README.md` section "Scope and limitations" before interpreting scores.
5. Ask only for decisions that cannot be learned from the checkout. Do not make
   the operator repeat an approval they already gave.

Prefer the current command help over memorized flags:

```bash
./venv/bin/python -m suite_tools.prepare_run --help
./venv/bin/python -m suite_tools.preflight_conditions --help
./venv/bin/python -m suite_tools.scheduler --help
./venv/bin/python -m suite_tools.bench --help
```

Use [`references/commands.md`](references/commands.md) for exact recipes.
Use [`../../docs/AGENT_RUNBOOK.md`](../../docs/AGENT_RUNBOOK.md) when the user
wants the complete fresh-clone workflow or is operating the repository through
an agent that did not invoke this skill by name.

## Rehydrate Before Acting

For continued work, first run:

```bash
./venv/bin/python -m suite_tools.companion resume --json
```

A missing active workflow is normal for a first visit; if the user expected to
resume something, run `./venv/bin/python -m suite_tools.companion list --json`
before starting a replacement. Otherwise, use the receipt to recover the
workflow goal, attached contracts, current phase, exact next action, choices,
and any still-valid approval. Then re-read the referenced
`RUN_CONTRACT.json`, `RUN_STATUS.json`, and `RUN_EVENTS.jsonl`; those scientific
artifacts always override companion state.

Use the companion for any workflow likely to cross messages, compaction, an
interruption, or a long run. It stores only prompt-free coordination metadata
under ignored `.benchmark-companion/`. It must never contain keys, prompts,
messages, transcripts, provider responses, or free-form private notes.

An approval is single-use and bound to the attached contract bytes, stage,
routes, and expected units. Record one only after explicit user approval.
Consume it immediately before launching that exact external operation; never
reuse it for another command or changed contract. Monitoring an already-running
operation does not need another approval. Read
[`references/companion.md`](references/companion.md) for the short command flow.

## Choose The Operating Mode

| Request | Mode | First action |
| --- | --- | --- |
| Check a clone or change | Offline verification | Validate config and run `offline_gate`. |
| Connect a model endpoint | Adapter onboarding | Prove the endpoint contract before a benchmark call. |
| Try one model cheaply | Paid smoke | Prepare one small contract, preflight it, then use `cautious`. |
| Collect comparable evidence | Official collection | Freeze independent contracts and verify every condition before spend. |
| Continue interrupted work | Recovery | Inspect owed units and failure class; resume the same contract only when valid. |
| Exclude a bad condition after completion | Derived subset | Materialize a new immutable AITA/EPIS subset; never edit the source run. |
| Package or share results | Evidence package | Clear the evidence-review gate, package, then verify the bundle. |

## Non-Negotiable Rules

- Use `suite_models.yaml`, `suite_tools.prepare_run`, and
  `suite_tools.scheduler` for current comparable runs.
- Treat `RUN_CONTRACT.json` as immutable. Never repair a run by editing its
  contract, transcripts, score files, receipts, hashes, or status ledger.
- Never use symlink staging or ad hoc shell backgrounding to create a subset or
  bypass scheduler limits.
- Keep private prompts, routing, keys, service IDs, and unpublished items in
  ignored private paths. Do not copy them into public artifacts.
- Capacity controls are operational only. They must not alter prompts,
  questions, request payloads, model IDs, judges, scoring, or promotion gates.
- Paces and leases limit concurrent calls, not dollars. Provider-side budgets
  are the hard spend control.
- Default judging sends target transcripts to the configured third-party judge
  providers. Confirm that data route is acceptable before paid scoring.
- Stop before spending when offline verification, exact-condition preflight,
  receipts, model locks, or expected-unit checks fail.

## The Normal Lifecycle

Follow one path: **declare -> preflight -> execute -> inspect -> score ->
review -> package**.

### 1. Declare

Validate the central registry, then prepare a no-paid contract. Use a registry
model key, comma-separated keys, `all`, or `group:<name>`; do not invent model
slugs. For official work, use separate child contracts for independently
restartable model, provider, or module groups.

Before accepting a prepared contract, inspect:

- model IDs, endpoints, efforts, and judge set;
- sample, item, scenario, side, and run counts;
- expected units and artifact paths;
- output-token controls and other explicit request options;
- `comparison_spec_hash`, model-condition hashes, and contract fingerprint;
- generated execute and score commands.

Do not assume a universal output cap such as 128k. Reasoning families may need
a uniform cap for comparability, but the correct value comes from current model
documentation and the explicit contract condition.

### 2. Preflight

Run the offline gate first. Its own documented skips are acceptable; any
unexpected failure or skip stops paid work.

For direct-provider, reasoning-effort, proxy, or newly added endpoint
conditions, run `preflight_conditions --run-dir <prepared-module-dir>` for every
child contract. This probes the exact model, effort, and endpoint cells. Every
target is a network request: bundled local reference-adapter probes are free;
remote or proxy probes may bill a small amount.

A registry catalog check is not a substitute for exact-condition preflight.

### 3. Execute

Use `scheduler run` for one contract or `scheduler run-many` for independent
contracts. Ask `scheduler paces` for the current presets rather than copying
old numbers.

`run-many` shares the global paid-call lease across children. Its
`--stop-on-attention` behavior stops the affected contract; it is not a
fleet-wide kill switch. Clean children continue, and clean generation is
scored as it finishes unless `--no-auto-score` is supplied.

For one official contract, keep generation and scoring as separate gates unless
the operator deliberately chose automatic scoring. Keep the dashboard open for
observation, but trust the ledger files rather than a visual tile alone.

Start the local dashboard before generation unless the user declines it. Give
the user the loopback URL and keep the dashboard process separate from the
scheduler. Opening the dashboard is local and free; it is not approval for a
preflight, generation call, or judge call.

### 4. Inspect

Use these files as the durable record:

- `RUN_STATUS.json`: current status, validity, counters, and failure reason;
- `RUN_EVENTS.jsonl`: append-only calls, writes, scores, receipts, and failures;
- `CALL_DIAGNOSTICS.jsonl`: private prompt-free call lifecycle and response-shape diagnostics;
- `RUN_CONTRACT.json`: expected work and immutable provenance;
- `SCHEDULER_STATUS.json` and `SCHEDULER_EVENTS.jsonl`: process-level state;
- `BLOCKS.jsonl` and `BLOCK_REVIEWS.jsonl`: evidence and review decisions.

Generation can be clean while the run is not yet score-ready. A normal scoring
gate is usually `status=completed` with `validity=not_score_ready`, shown as
"Needs Scoring." Final publishable state is `status=completed` and
`validity=score_ready`.

Do not infer health from a `running` label. Confirm recent writes, a live
scheduler process, expected spend movement, and the absence of a terminal
failure.

If an error is unclear or a call may have died between invocation and receipt,
run `bench diagnose <run_dir> --json`. An attempt that reached
`provider_invocation_started` but not `closed` is billing-ambiguous; check the
provider's usage before retrying. The diagnostic journal is operational only:
it is excluded from packages and does not change contract or comparison hashes.

### 5. Score

Score only clean, complete generation. Effective-request receipts must match
the request controls frozen in the contract. Do not synthesize or backfill
receipts to rescue a run.

Before any judge call, require `bench verify <run_dir> --json` to report both
`request_conformance.conformant=true` and
`artifact_identity.conformant=true`. A completed transcript is reusable only
when its saved `condition_id` and `condition_hash` agree with the same rendered
condition named by the frozen contract.

Also run `./venv/bin/python -m suite_tools.hygiene_gate <run_dir>` before scoring or
packaging. It is offline and observation-only; a non-zero exit means saved
transcripts contain blocking error text, empty responses, malformed wrappers,
or incomplete conversations that must not be scored as model behavior.

Retry scoring without regeneration only when generation is complete and the
failure belongs to the judge or score stage. Use `--force` only where current
CLI help and `RUNBOOK.md` allow it.

### 6. Evidence Package

Use `bench review` to disposition unresolved evidence. `bench package` is
fail-closed: incomplete members, unresolved facts, active escalation reviews,
non-publishable winning units, or fingerprint drift block emission. Always run
`bench verify --bundle` on the emitted bundle.

The benchmark stops here; manuscript writing is downstream and outside this
skill.

## Adapter Onboarding

If a system already serves OpenAI Chat Completions, configure it as an
`openai_compatible` endpoint. Do not write a custom runner. Use a private
registry overlay for private URLs, credential names, and internal routing
labels. Public bundles intentionally retain the tested model key, model ID,
display label, and condition hashes, so use publication-safe aliases for those
fields before preparing anything you may package. An existing endpoint needs
the chat-completions boundary; it does not need the bundled adapter's `/health`
or `/v1/models` routes. Prove it with `preflight_conditions` against the
prepared condition.

If the backend is not OpenAI-compatible, adapt only the backend seam in the
bundled reference adapter. Keep the inbound `/v1/models`,
`/v1/chat/completions`, response text, usage, refusal-only normalization, and
structured error behavior intact.

When using the bundled adapter, prove layers in this order:

1. endpoint health and model discovery;
2. one deterministic `adapter/smoke.py` response;
3. exact-condition preflight;
4. one prepared SUS smoke contract;
5. generation inspection;
6. scoring with an approved judge route.

The smoke tool refuses proxy calls unless `--allow-proxy-call` is explicit.
Read [`adapter/README.md`](../../adapter/README.md) for the request and response
contract, reference mode, proxy mode, and private-backend example.

## AITA N=20 Add-On

Before preparing the full AITA condition, run
`./venv/bin/python -m suite_tools.aita_data_pack status --json`. The command is
local and network-free. If `download_available` is false, state that the pack
is not available from this release and do not invent a URL, reuse development
plaintext, or substitute the synthetic smoke fixture.

When the signed data release is available, show the user the status receipt's
repository, release, exact asset URLs, envelope and ciphertext file names, byte
counts, hashes, exact Part B suite-release asset URL, and local destination. Require
`run_available: true`, then ask for approval before downloading.
After approval, run `suite_tools.aita_data_pack fetch` with
`--confirm-download`, then require its verified receipt before preparation.
The registry hashes are frozen into the authenticated software release. A data
download is not a provider call, but it is still an external action and must be
visible to the user. Never add `--confirm-download` before the user approves.

Tell the user before preparation that the CLI will visibly prompt for Part B
and hide only the typed characters. Preparation and generation each reacquire
Part B. Never print it, put it in chat or a command argument, or store it in
companion state. Do not use the noninteractive environment route unless the
user explicitly accepts its warning and local secret-handling tradeoff.

## Recovery Decisions

Classify the failure before rerunning:

| Signal | Meaning | Action |
| --- | --- | --- |
| `insufficient_quota` | Account credits exhausted | Refill, clear stale control if needed, then resume the same contract. |
| Rate-limit 429 or provider 5xx | Transient environment failure | Let bounded retries work; resume owed units if the attempt ended. |
| Content-policy refusal | Terminal model/provider signal | Preserve it for review; do not disguise it as infrastructure failure. |
| Malformed payload, missing receipt, adapter mismatch | Instrument defect | Stop, fix the harness, and regenerate affected work in a fresh contract when required. |
| Judge/schema failure after clean generation | Scoring failure | Repair the score path and rescore existing complete transcripts. |

Preview owed work before resuming. Reuse only artifacts the runner recognizes as
complete and whose condition identity matches the same rendered condition. A
runner may restore a missing identity field in memory from that one frozen
condition, but any conflict is an instrument defect and must stop before paid
work. Clear `RUN_CONTROL.json` through `scheduler clear-control`; never delete
or rewrite evidence to make the run look clean.

Pre-receipt direct-OpenAI work with explicit request controls is not verifiable
and the affected conditions cannot be repaired or retained. When they are part
of a mixed AITA or Epistemic source that is already completed and
`score_ready`, `materialize_subset` can preserve only the unaffected conditions
in a new immutable run. Exclude every unverifiable model key. This is an
offline exclusion workflow, not regeneration or a way to bless partial work.
For a saved SUS rescore, merge, or serialization defect where missing identity
is recoverable from exactly one frozen source-contract condition,
`materialize_sus_derived` may emit a new no-provider derived run with source
hashes and an explicit normalization receipt. It never edits the source and it
must reject conflicting or ambiguous identity.

## Report Results Honestly

Never present a bare score or ranking. Carry these limitations:

- SUS currently measures one scenario, `bridge_heights`.
- The LLM judges lack a human-labeled validation set and agreement statistic.
- At n=20, Wilson intervals are wide enough that overlapping results are not a
  meaningful ranking.
- Target temperature is generally provider-default and unpinned.
- Blanket refusal can score well, so low capitulation is not proof of
  helpfulness.
- Model and judge aliases can move over time; report the run date and hashes.

See `docs/HARDENING_BACKLOG.md` for the maintained methodology gaps.

## Stop Conditions

Pause before the next paid call when any of these is true:

- the requested scope, model condition, judge route, or spend approval is
  genuinely ambiguous;
- config validation or the offline gate is not clean;
- exact-condition preflight does not pass;
- expected units, request controls, receipts, or hashes disagree;
- provider credits, status, or recent errors make continued spend unsafe;
- the dashboard and ledgers disagree and the discrepancy is unexplained.

When blocked, report the exact failed gate, what evidence remains valid, what is
owed, and the smallest scientifically defensible next step.
