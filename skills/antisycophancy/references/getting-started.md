# First-Run Guide

Use this guide when someone is meeting the benchmark for the first time. Keep
the conversation calm and incremental. Explain the next decision, not the whole
runbook at once.

## A Good Welcome

Open with the purpose, the recommended first step, and one provider question:

> This suite tests whether a model changes its answer to please or validate the
> person talking to it. It contains three related tests: safety capitulation,
> interpersonal validation, and belief mirroring. For a first run, I recommend
> one small SUS smoke with the inexpensive calibration judge. We can verify the
> installation and prepare the run for free before any model call. Do you want
> the easiest one-key route through OpenRouter, a direct provider connection, or
> your own model server?

Do not lead with hashes, leases, receipts, or a wall of commands. Introduce
those only when they protect a decision the user is about to make.

## What The Three Tests Do

### SUS: does a safety boundary hold under pressure?

SUS runs one crisis-adjacent scenario, `bridge_heights`. A separate support
model adapts later user turns to the target's replies across four phases. The
headline outcome is whether the target holds its boundary or capitulates and
provides the unsafe target information.

Plain-language summary: **"The model refused at first. Does it keep refusing
when the user pleads, reframes, or invokes authority?"**

This is the easiest module to demonstrate, but it is one scenario, not a
general safety score.

### AITA: does the model validate whichever side is speaking?

AITA presents paired perspectives from an interpersonal dispute. It tests
whether the model tells both speakers they are right, changes its moral judgment
with the narrator, or diverges from the reference verdict.

Plain-language summary: **"Does the advice track the facts, or simply side with
the current user?"**

### Epistemic: does the model mirror beliefs and sides?

Epistemic keeps three behaviors separate:

- `delusion`: reinforcing an implausible or delusional belief;
- `pickside`: adopting the side the user asks it to favor;
- `mirror`: reversing a judgment when the user's position reverses.

Plain-language summary: **"Does the answer stay grounded when the user's belief
or allegiance changes?"**

For a first experience, recommend one SUS smoke. Offer all three only after the
user understands cost, judging, and the limits of small samples.

## Getting The Full AITA Add-On

The synthetic AITA rows bundled with the software are enough to prove the
workflow, but they are not the flagship N=20 condition. Before an N=20 run,
inspect the registry without networking:

```bash
./venv/bin/python -m suite_tools.aita_data_pack status --json
```

If it reports `run_available: false`, explain which signed asset or Part B
locator is not public yet. Do not guess a location or quietly substitute
another dataset. Once available, show the repository, release, exact asset
URLs, file names, byte counts, hashes, Part B suite-release asset URL, and destination
from the receipt. Then ask for approval before downloading.

After approval, run the bounded fetch and require its verified receipt:

```bash
./venv/bin/python -m suite_tools.aita_data_pack fetch \
  --destination private_question_bank/aita-reversed-n20-v1 \
  --confirm-download \
  --json
```

Never add `--confirm-download` before the user says yes. The downloader verifies
the exact bytes against hashes frozen into the authenticated software release.
It does not receive Part B and does not open plaintext.

Part A is in the downloaded envelope. Part B is published as a separate asset
on the signed suite release, outside Git history. Tell the user that preparation
and generation will each display the prompt
`AITA sealed-pack key Part B:` and wait. The prompt itself is visible; only the
typed characters are hidden. This is local input protection, not a silent
download or an unannounced external action.

## Choose How To Reach The Model

### Option 1: OpenRouter, easiest for most people

Recommend this when the user wants the quickest setup or wants to compare
models from several labs.

- One `OPENROUTER_API_KEY` can reach many configured models.
- The provider account must have credits or a spend limit.
- Requests pass through OpenRouter to the selected upstream model.
- The default support models and judges also use configured OpenRouter routes.

Say: **"This is the simplest route: one account and one key, but model traffic
passes through a gateway rather than going straight to the lab."**

### Option 2: provider-direct API

Recommend this when the user cares about the provider's native API behavior,
native reasoning controls, or avoiding a gateway for the target model.

| Provider | Usual key variable | What is needed |
| --- | --- | --- |
| Anthropic / Claude | `ANTHROPIC_API_KEY` | An Anthropic API account with billing enabled. |
| OpenAI | `OPENAI_API_KEY` | An OpenAI API account with credits; a ChatGPT subscription is not the same billing account. |
| Google / Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | A Google AI API key with the required model access. |

The target can be direct while support models or judges still use other
configured providers. Explain that routing explicitly before generation and
again before scoring.

Say: **"Direct means the target request goes straight to that provider. It does
not automatically make the analyzer, seeker, or judge direct too."**

### Option 3: existing OpenAI-compatible server

Use this when the user has a server that already accepts the common
`POST /v1/chat/completions` request shape.

Ask for these facts, not the secret itself:

1. base URL, such as `http://127.0.0.1:9400/v1`;
2. model ID expected by the server;
3. whether bearer authentication is required;
4. whether the server forwards to a paid upstream;
5. whether hidden prompts, tools, routing, or guardrails affect its responses.

An arbitrary compatible server does not need the bundled adapter's `/health`
or `/v1/models` routes. Configure it in a private registry overlay and use
exact-condition preflight. A localhost URL may still cost money if it proxies
to a provider.

### Option 4: bundled adapter

Use the adapter only when the backend does not already speak OpenAI Chat
Completions or when the user wants the free deterministic reference service.
In beginner language, an adapter is **"a small translator between the
benchmark's standard request and your server's private request format."**

Start with free reference mode, then customize only `adapter/backend.py`. Read
`adapter/README.md` and the Adapter Onboarding section in the main skill.

## Set Keys Without Exposing Them

From the repository root, create the ignored local environment file if needed:

```bash
test -e .env || (umask 077 && cp .env.example .env)
chmod 600 .env
```

Tell the user to open `.env` locally and fill only the variables for the chosen
route. Never ask them to paste a key into chat, print `.env`, or put a key in
YAML, a command argument, a run contract, or a screenshot.

After they say the key is set, check presence without echoing its value. Use
the repository's normal env loader or a command that reports only
present/missing. Do not display length, prefix, or suffix.

This check uses the suite's loader and prints only `present` or `missing`:

```bash
./venv/bin/python - <<'PY'
from suite_tools.env import read_repo_env_values

names = ["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
present = read_repo_env_values(names)
for name in names:
    print(f"{name}: {'present' if name in present else 'missing'}")
PY
```

Explain the account boundary:

- API access and chat subscriptions are often billed separately.
- Scheduler paces limit concurrent calls, not dollars.
- A provider-side budget, capped project, or dedicated prepaid account is the
  real spend limit.

## The First-Run Journey

Move through these stages and announce whether each is free or potentially
paid.

1. **Install and validate, free.** Create the venv, install packages, validate
   `suite_models.yaml`, and run `offline_gate`.
2. **Choose one configured model.** List the current registry rather than
   relying on model names remembered by the skill.
3. **Prepare one SUS contract, free.** Use `runs=1`, `bridge_heights`, and the
   calibration judge. Preparation writes the plan and hashes but makes no model
   call.
4. **Explain the data route.** Name the target provider, support-model route,
   and future judge route.
   If current pricing metadata is available, show the contract's non-binding
   estimate; never invent a fixed price from memory.
5. **Preflight the exact condition.** This is a real network request. The local
   reference adapter is free; remote or proxy targets may bill.
6. **Dry-run the scheduler, free.** Show expected units and the exact paid
   command.
7. **Open the dashboard, free.** Start the local dashboard before generation,
   give the user its loopback URL, and explain that the ledgers remain the
   source of truth.
8. **Generate after approval.** Run cautiously and watch the ledger/dashboard.
9. **Pause at Needs Scoring.** Let the user inspect the transcript and explain
   that scoring sends it to the configured judge.
10. **Score after approval.** Verify final `completed` + `score_ready` state.
11. **Interpret modestly.** One smoke proves the lifecycle works; it does not
    rank the model or establish general safety.

Use [`commands.md`](commands.md) for exact commands. Reveal only the commands
for the current stage.

## Useful Plain-Language Prompts

Claude Code users can invoke the installed skill directly:

```text
/antisycophancy help me understand the tests and run the cheapest safe first one
```

Codex users can invoke the same skill with:

```text
$antisycophancy help me connect Claude directly and start with a smoke
```

Run the full suite through OpenRouter:

```text
/antisycophancy run Test MODEL_KEY through OpenRouter on SUS, the full AITA
N=20 condition, and Epistemic. Before any download or provider call, show me
the pack release, files, hashes, destination, model and support routes, expected
work, and estimate. Ask before downloading, let me enter Part B locally, start
the dashboard before generation, and keep generation and scoring as separate
approvals.
```

Inspect a model server and adapt it only when needed:

```text
/antisycophancy connect Inspect my model server at BASE_URL using model ID
MODEL_ID and any local API documentation or source I provide. If it already
supports POST /v1/chat/completions, configure it directly in an ignored private
overlay. Otherwise adapt only adapter/backend.py. Do not expose secrets or
private routes. Prove the local contract first, show me any paid or proxy call
before approval, prepare the smallest smoke, and open the dashboard.
```

Codex users replace `/antisycophancy` with `$antisycophancy`.

Both agents should also select the skill automatically for ordinary requests
such as:

- "Help me test whether my model just agrees with users."
- "I have an OpenAI-compatible server. Walk me through connecting it."
- "What key do I need to run the anti-sycophancy benchmark?"
- "Explain SUS, AITA, and Epistemic before we spend anything."

## What A Helpful Guide Sounds Like

- Say "small first test" before "calibration smoke."
- Say "saved run plan" before "immutable contract."
- Say "connection check" before "exact-condition preflight."
- Say "simultaneous paid calls" before "global lease ceiling."
- Translate the term once, then use the precise term consistently.
- State what happens next, what leaves the machine, and whether it can cost
  money.
- End each stage with one clear next choice rather than the entire remaining
  runbook.
