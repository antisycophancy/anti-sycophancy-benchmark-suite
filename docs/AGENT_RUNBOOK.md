# Agent Runbook

This guide is for a person opening the repository with Claude Code, Codex, or
another coding agent. It gives the agent one safe path from a fresh clone to a
verified result without depending on chat history or memorized command flags.

The agent should also read [`../AGENTS.md`](../AGENTS.md) and
[`../skills/antisycophancy/SKILL.md`](../skills/antisycophancy/SKILL.md). This
runbook explains the whole journey in one place. The skill provides the detailed
decision rules and exact command reference.

## Start here

After bootstrap, open the repository in Claude Code or Codex. The repository
already includes the skill and the agent instructions it needs.

In Claude Code, start with a normal request.

```text
/antisycophancy Help me choose a benchmark and connect OpenRouter. Prepare the
smallest useful smoke test, show me the estimate, and open the dashboard.
```

In Codex, use the same request with `$antisycophancy`. You can also ask in plain
language without naming the skill.

If you already know what you want to connect, say so directly.

```text
/antisycophancy Connect my OpenAI-compatible model at
http://127.0.0.1:9400/v1. The model ID is my-model. Verify the connection, then
help me choose a benchmark.
```

```text
/antisycophancy Resume my last benchmark run and tell me what is left.
```

The agent handles the safety details behind those requests. Before an external
or paid step, it should show the selected models, routes, support roles, call
count, estimate, data route, and exact command. Generation and scoring remain
separate decisions unless you ask to combine them for the prepared run.

## What the agent must protect

The benchmark separates scientific identity from operator convenience. The
agent must preserve that boundary.

- Never ask the user to paste a key into chat.
- Never put a key in YAML, a command argument, a contract, or a result file.
- Never make a remote preflight, generation call, or judge call without approval
  for that exact scope.
- Never edit a prepared `RUN_CONTRACT.json` or completed run artifact.
- Never treat a dashboard tile as stronger evidence than the ledger files.
- Never rerun a billing-ambiguous attempt until provider usage is checked.
- Never package a run that fails request, artifact, hygiene, evidence, privacy,
  or provenance checks.
- Never copy private URLs, service IDs, prompts, or routing details into public
  model labels or bundle metadata.

## The first five minutes

The agent should perform these steps before recommending a run.

1. Confirm the repository root and read `AGENTS.md`.
2. Inspect `git status --short` and preserve unrelated work.
3. Run the prompt-free companion resume command.
4. Confirm that the source has been authenticated and bootstrap completed.
5. Run offline configuration and test checks.

```bash
./venv/bin/python -m suite_tools.companion resume --json
./venv/bin/python -m suite_tools.model_config --validate
./venv/bin/python -m suite_tools.offline_gate
```

`companion resume` returning "No active companion workflow" is normal for a
new user. An offline failure stops the workflow. A skip is acceptable only when
its printed reason names a documented optional or private check.

## Explain the three choices

The agent should explain the modules in ordinary language before asking the
user to choose.

| Module | Plain-language question | Good first use |
| --- | --- | --- |
| SUS | Will the model keep a safety boundary when the user keeps reframing the request? | Smallest operational smoke and safety-boundary testing. |
| AITA | Will the model validate both sides of the same social conflict? | Paired social-consistency testing with the separately distributed sealed pack. |
| Epistemic | Will the model mirror beliefs or change its assessment when the user's side changes? | Belief, delusion, and side-swap behavior. |

For a new operator, recommend one SUS scenario, one model, one run, and the
single calibration judge. It exercises preparation, preflight, generation,
verification, scoring, and the dashboard with the smallest useful scope.

## Choose the connection

Ask only for facts that are not available in the repository. Do not ask for the
secret value.

| Route | Facts to confirm |
| --- | --- |
| OpenRouter | The key variable exists, the account has a spending limit, and gateway routing is acceptable. |
| Provider direct | The provider, native API family, model access, and target key variable. Support roles may still use other providers. |
| Existing compatible server | Base URL, model ID, key variable name, whether the server proxies to a paid upstream, and whether hidden prompts or routing affect behavior. |
| Bundled adapter | Whether the user wants free reference mode or needs to translate a proprietary backend. |

List the current registry rather than relying on model names from an old chat.

```bash
./venv/bin/python -m suite_tools.model_config --list --output-json
```

For a private model, create an ignored overlay under `private_profiles/`. Give
the model a publication-safe key, model ID, and label before preparing a run
that may be shared. The public bundle can retain those identifiers even though
it removes private routes and credential names.

## Set keys locally

The supported key file is the repository-root `.env`. Create it with owner-only
permissions and do not overwrite an existing file.

```bash
test -e .env || (umask 077 && cp .env.example .env)
chmod 600 .env
```

Common variables include `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, and `GEMINI_API_KEY`. A custom endpoint should use a dedicated
variable such as `MY_ENDPOINT_API_KEY`. Missing custom credentials must fail
closed and must never fall back to the OpenRouter key.

The agent may check whether a variable is present. It must not print its value.

## Capture current pricing

For an OpenRouter run, capture the current catalog and pricing data before
preparation. This is a network metadata request, not a model generation call.

```bash
mkdir -p results/operator
./venv/bin/python -m suite_tools.openrouter_preflight \
  --config suite_models.yaml \
  --strict-pricing --json \
  > results/operator/openrouter-pricing.json
```

The estimate is a planning aid. It is calculated from the frozen call plan and
the available uncached input, cached input, output, and reasoning-token prices.
Missing prices remain unknown. The provider account limit is the hard spending
control.

## Prepare without spending

Preparation writes a run group and one module contract. It makes no provider
call.

```bash
./venv/bin/python -m suite_tools.prepare_run \
  --module sus \
  --run-id first-smoke \
  --models group:calibration_smoke \
  --judge-set calibration \
  --scenarios bridge_heights \
  --runs 1 \
  --pricing-snapshot results/operator/openrouter-pricing.json \
  --warn-above-usd 1.00 \
  --output results/prepared/first-smoke \
  --non-interactive --output-json
```

Before asking for preflight approval, the agent should summarize these fields
from the generated files.

- Module, run ID, lifecycle state, and contract fingerprint
- Tested model key, model ID, provider API, endpoint class, and request controls
- Support agent and judge routes
- Scenario or item count and expected units
- Call-plan range and estimated cost range
- Unknown pricing or usage assumptions
- Generated execution and score commands
- Public identifiers and private data routes

Use scheduler dry-run when the user wants to inspect command admission without
spending.

```bash
./venv/bin/python -m suite_tools.scheduler run \
  --contract results/prepared/first-smoke/sus/RUN_CONTRACT.json \
  --dry-run --output-json
```

Dry-run does not prove a provider condition and does not create an accepted
preflight receipt.

## Preflight the exact condition

Prepared-run preflight covers the model under test and every paid support role
rendered into the contract. It uses the actual request controls with a small
output cap and creates `PREFLIGHT_RECEIPT.json`.

```bash
./venv/bin/python -m suite_tools.preflight_conditions \
  --run-dir results/prepared/first-smoke/sus \
  --json
```

Every target is a real network request. A local reference-adapter request is
free. A remote or proxy request may be billed. Ask for approval after showing
the target count and routes. Any non-PASS result stops execution.

The receipt expires after six hours. The scheduler recomputes the exact target
set and reauthenticates the prepared config before trusting it.

## Start the dashboard

Run the dashboard in its own terminal and leave the scheduler in another. Do
not hide a long run behind ad hoc shell backgrounding.

```bash
./venv/bin/python -m suite_tools.live_dashboard \
  --results-root results/prepared \
  --port 8765 \
  --operator-id local:your-name
```

Open `http://127.0.0.1:8765`. The dashboard is loopback-only by default. It
reads the same contracts, status files, events, diagnostics, and review
sidecars that the CLI uses.

## Run generation

Ask for generation approval separately from scoring unless the user explicitly
approved automatic scoring for this contract. The safest first run stops at the
generation gate.

```bash
./venv/bin/python -m suite_tools.scheduler run \
  --contract results/prepared/first-smoke/sus/RUN_CONTRACT.json \
  --run-pace cautious \
  --stop-on-attention \
  --gate-after-generation
```

The scheduler and model registry jointly limit concurrency. A pace limits
simultaneous calls. It does not cap dollars. Do not raise
`--max-active-calls` without checking the registry limits, `.env` policy,
provider limits, and current account budget.

## Verify before scoring

Generation completion is not enough. The agent should require artifact
identity, request conformance, and transcript hygiene before asking a judge to
read the responses.

```bash
./venv/bin/python -m suite_tools.bench verify \
  results/prepared/first-smoke/sus --json

./venv/bin/python -m suite_tools.hygiene_gate \
  results/prepared/first-smoke/sus
```

The verification result must report conformant request and artifact identity.
The hygiene gate must exit zero. Do not edit transcripts or synthesize receipts
to make a failed run pass.

## Run scoring

Before asking for scoring approval, name every judge route and explain that the
saved target transcript will be sent to those providers. Refresh exact
preflight if the receipt has expired.

```bash
./venv/bin/python -m suite_tools.scheduler score \
  --contract results/prepared/first-smoke/sus/RUN_CONTRACT.json \
  --run-pace cautious \
  --stop-on-attention
```

The run is ready for publication review only when `RUN_STATUS.json` reports
`status=completed` and `validity=score_ready`.

## Review and package

Inspect unresolved evidence before adopting a run into an experiment.

```bash
./venv/bin/python -m suite_tools.bench review --json
./venv/bin/python -m suite_tools.bench verify <run-dir> --json
```

Use `bench adopt` and `bench package` only after the user identifies the
experiment and publication role. Packaging refuses incomplete, unresolved,
non-publishable, drifting, or privacy-unsafe evidence. Verify the emitted
bundle separately.

```bash
./venv/bin/python -m suite_tools.bench package \
  <experiment-dir> --out results/bundles --json

./venv/bin/python -m suite_tools.bench verify \
  --bundle <bundle-dir> --json
```

## Connect an existing endpoint

An existing compatible endpoint needs `POST /v1/chat/completions`. It does not
need the bundled adapter's health or model-list routes.

Create a private overlay, set its dedicated key variable in `.env`, validate
the overlay, and prepare with that same `--config` path. For an intentional
remote HTTPS endpoint, authorize its exact hostname through
`BENCHMARK_ALLOWED_ENDPOINT_HOSTS`. Official provider key variables remain
bound to their official origins.

Read [`../skills/antisycophancy/references/commands.md`](../skills/antisycophancy/references/commands.md)
for the complete overlay example and commands.

## Use the reference adapter

If the backend does not accept Chat Completions, use the canonical adapter.
Start with free reference mode.

```bash
test -e adapter/.env || (umask 077 && cp adapter/.env.example adapter/.env)
chmod 600 adapter/.env
./venv/bin/python adapter/server.py
```

From another terminal, prove the adapter contract.

```bash
./venv/bin/python adapter/smoke.py
```

Proxy mode requires inbound authentication even on loopback. Its smoke refuses
the chat request unless `--allow-proxy-call` is explicit. A local proxy may
still bill a remote provider.

Customize only `adapter/backend.py` for a proprietary request or response
shape. Keep the public server, authentication, input limits, structured errors,
and benchmark-facing response contract unchanged. Read
[`../adapter/README.md`](../adapter/README.md) before making that change.

## Run AITA with the sealed pack

AITA N=20 needs the separately signed pack named by
`manifests/aita-sealed-pack-v1.json`. Inspect its availability without
networking.

```bash
./venv/bin/python -m suite_tools.aita_data_pack status --json
```

The agent must not guess a download location. If `download_available` is false,
tell the user that the separate data release is not public yet and stop the
N=20 path. Once available, show the receipt's repository, release, exact asset
URLs, file names, byte counts, hashes, and local destination. Ask before
downloading. After approval, use the bounded fetch and require its verified
receipt before preparation.

```bash
./venv/bin/python -m suite_tools.aita_data_pack fetch \
  --destination private_question_bank/aita-reversed-n20-v1 \
  --confirm-download \
  --json
```

Never add `--confirm-download` before the user approves. The downloader checks
the exact bytes against hashes frozen into the authenticated software release.
It does not receive Part B and does not open plaintext.

Hidden input is not a hidden action. Tell the user that preparation and
generation each display `AITA sealed-pack key Part B:` and wait. The prompt is
visible while the characters they type are not. Entering the fragment is local
and does not itself make a network call. Never print, log, or store that
fragment in companion state.

Pass the envelope with `--sealed-pack`. Do not mix it with plaintext AITA data
arguments. Preparation and runtime each authenticate the exact pack. Public
packaging refuses raw sealed-run transcripts and free-text review rationale.

## Resume safely

Use the companion to recover the workflow intent, then use the run ledgers for
scientific truth.

```bash
./venv/bin/python -m suite_tools.companion resume --json
./venv/bin/python -m suite_tools.bench status <run-dir> --json
./venv/bin/python -m suite_tools.bench diagnose <run-dir> --json
```

Classify the failure before continuing.

| Signal | Response |
| --- | --- |
| Insufficient quota | Refill the provider account, inspect control state, and resume owed units under the same contract. |
| Rate limit or provider 5xx | Let bounded retries finish, then resume only owed work. |
| Content-policy refusal | Preserve it as model or provider evidence. Do not relabel it as infrastructure failure. |
| Missing receipt, malformed payload, or adapter mismatch | Stop for an instrument fix. Use a fresh contract when affected generation cannot remain comparable. |
| Judge or score failure after clean generation | Repair the score path and rescore the existing complete transcripts. |
| Invocation started without a closing receipt | Treat billing as ambiguous and check provider usage before retrying. |

Use `scheduler stop` for cooperative stop and `scheduler clear-control` for a
settled stale control file. Never delete evidence to make a run appear clean.

## What the agent should report

At each boundary, the agent should give the user a short operational receipt.

- What was checked or changed
- What is proven and what is still unverified
- Which files carry the durable evidence
- Which routes and providers would receive data next
- The estimated range and every unknown cost component
- Whether the next command is offline, network-only, generation, or scoring
- The exact approval needed, if any

At completion, report the module, contract fingerprint, model conditions,
sample size, score directions, judge panel, verification result, bundle state,
cost provenance, and the limitations from the main README. Do not turn one
small run into a broad model claim.
