# Resumable Companion

Use this for work that may continue across messages, compaction, interruption,
or a long provider run. It adds no service or dependency. The normal editable
installation already includes `suite_tools.companion`.

## Truth Boundary

- Benchmark contracts, statuses, events, receipts, and reviews say what
  scientifically happened.
- `.benchmark-companion/` remembers which contracts belong to the workflow,
  the broad goal, exact approvals, one-time approval claims, and the next action
  derived from the current ledgers.
- `RESUME.json` is a cache, not authority. It is rebuilt on every resume.

The event log never stores keys, prompts, messages, transcripts, provider
responses, or free-form notes. Deleting the companion directory loses
coordination convenience but does not remove or alter benchmark evidence.

## Begin Or Resume

Start a workflow before preparing a contract, or attach the contract at start:

```bash
./venv/bin/python -m suite_tools.companion start WORKFLOW \
  --goal onboarding --json

./venv/bin/python -m suite_tools.companion start WORKFLOW \
  --goal collection --run results/prepared/RUN_ID/sus --json
```

During onboarding, preserve only the small allowlisted choices that would
otherwise be lost across compaction:

```bash
./venv/bin/python -m suite_tools.companion choose WORKFLOW \
  --key connection_route --value provider_direct --json
./venv/bin/python -m suite_tools.companion choose WORKFLOW \
  --key target_provider --value anthropic --json
```

Use `choose --help` for allowed keys and values. Do not encode arbitrary notes
or model prompts in workflow IDs or choices.

After preparing another child contract:

```bash
./venv/bin/python -m suite_tools.companion attach WORKFLOW \
  --run results/prepared/RUN_ID/aita --json
```

At the start of every later invocation, omit the ID to resume the active one:

```bash
./venv/bin/python -m suite_tools.companion resume --json
```

The receipt stops on ledger contradictions rather than guessing. `list --json`
shows all local workflows without changing which one is active.

## Paid Boundaries

Only after the user explicitly approves the displayed contract, route, stage,
and expected units, record the approval:

```bash
./venv/bin/python -m suite_tools.companion approve WORKFLOW \
  --run results/prepared/RUN_ID/sus \
  --stage generation --confirmed-by-user --json
```

Immediately before launching that operation, consume its returned approval ID:

```bash
./venv/bin/python -m suite_tools.companion consume WORKFLOW \
  --approval APPROVAL_ID --json
```

Then launch the exact preflight, generation, or scoring command. Consumption
before execution prevents an interrupted agent from accidentally spending the
same permission twice. If the command never launches, ask again rather than
reactivating the old approval. Contract or route drift invalidates an
unconsumed approval automatically.

Run `./venv/bin/python -m suite_tools.companion --help` for the complete local CLI.
