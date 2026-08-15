# Benchmark Calibration Runbook

> **Relationship to RUNBOOK.md:** RUNBOOK.md is the primary operations reference
> (provider setup, prepared contracts, scheduling, scoring). This file covers
> the cheaper, pre-spend workflow: smoke runs, wiring checks, judge calibration,
> and transcript review before committing to a full frontier run.

This runbook covers how to check wiring, judge quality, and adaptive prompt behavior without paying for full frontier runs.

## Goals

- Keep model, endpoint, judge, seeker, and flip-generator choices in one place: `suite_models.yaml`.
- Render module-native configs only when needed.
- Run tiny smoke/calibration samples before any full paid run.
- Preserve conversations so we can manually read whether the adaptive pressure is testing the intended thing.
- Re-score existing conversations with stronger judges whenever possible instead of re-running model-under-test calls.

## Central Config

Edit `suite_models.yaml`, then render native config files:

```bash
cd /path/to/benchmark
./venv/bin/python -m suite_tools.model_config --validate
./venv/bin/python -m suite_tools.model_config --list
./venv/bin/python -m suite_tools.model_config --list --output-json
python3 -m suite_tools.model_config \
  --judge-set calibration \
  --models group:calibration_smoke \
  --output-dir /tmp/benchmark-configs
```

`suite_models.yaml` supports OpenAI-compatible endpoints with separate
`openai_base_url`, `chat_completions_url`, and `api_key_env` fields. Add new
raw models by adding one `models.<key>` entry and, when useful, a
`model_groups.<group>` entry. Add a new endpoint only when the provider exposes
an OpenAI-compatible API surface; native Anthropic/Gemini SDK support should be
handled as a separate adapter layer rather than mixed into benchmark runners.

For stronger judge scoring:

```bash
python3 -m suite_tools.model_config \
  --judge-set frontier \
  --models group:frontier_03_04 \
  --output-dir /tmp/benchmark-configs
```

Generated configs are disposable and ignored by git.

## Manual Transcript Review

Render any saved run into readable Markdown:

```bash
python3 -m suite_tools.inspect_conversations \
  sus-bench/results/sus-bench-YYYYMMDD-HHMMSS-conversations.json \
  --limit 3 \
  --output /tmp/sus-review.md
```

Use the same helper on AITA or Epistemic run directories:

```bash
python3 -m suite_tools.inspect_conversations results/aita/aita-smoke --limit 4
python3 -m suite_tools.inspect_conversations epistemic-sycophancy-bench/results/epis-smoke --limit 6
```

Build a local HTML viewer when you need to compare full transcripts against
judge scores:

```bash
python3 -m suite_tools.review_viewer \
  sus-bench/results/sus-bench-YYYYMMDD-HHMMSS-conversations.json \
  results/aita/aita-smoke \
  epistemic-sycophancy-bench/results/epis-smoke \
  --output /tmp/benchmark-review.html
```

The viewer is static HTML. It embeds source paths, paired score JSON, judge
model metadata, and full conversations so a human can spot-check whether the
judge scores match the transcript before scaling up.

## Live Run Dashboard

Use the local live dashboard when a paid smoke or official run is in progress:

```bash
cd /path/to/benchmark
./venv/bin/python -m suite_tools.live_dashboard \
  --results-root results/testing \
  --port 8765
```

Open `http://127.0.0.1:8765`. The dashboard is read-only: it polls
`RUN_CONTRACT.json`, `RUN_STATUS.json`, `RUN_EVENTS.jsonl`, and optional
`SCHEDULER_STATUS.json` / `SCHEDULER_EVENTS.jsonl` files. It groups modules by
run directory, shows elapsed time, status, validity, expected work, queued
scheduler state, ETA from completed-unit averages, turns saved, scoring
progress, failures, control state, and any cost data written by the runners. It
also surfaces model locks, judge metadata, a benchmark comparison identity
panel, and spend guards from tracked cost/credit data. Terminal or scheduler
commands remain the source of truth for launching and stopping runs.
If port `8765` is already occupied, rerun with a different `--port`.

For machine-readable inspection without serving HTML:

```bash
./venv/bin/python -m suite_tools.live_dashboard \
  --results-root results/testing \
  --once
```

## Production Run Integrity Ledger

Paid runs now write a small machine-readable run ledger next to their artifacts:

```text
RUN_CONTRACT.json # expected work, models, judges, artifacts, gates
RUN_CONTROL.json  # optional cooperative stop/pause intent
RUN_STATUS.json   # latest stage, validity, counters, and abort reason
RUN_EVENTS.jsonl  # append-only event stream for turns, scores, and failures
SCHEDULER_STATUS.json  # optional process scheduler state and ETA
SCHEDULER_EVENTS.jsonl # optional process scheduler events
```

For a prepared contract, dry-run or execute through the scheduler:

```bash
./venv/bin/python -m suite_tools.prepare_run \
  --module sus \
  --run-id RUN_ID \
  --models group:calibration_smoke \
  --judge-set calibration \
  --scenarios bridge_heights \
  --runs 1 \
  --output results/prepared/RUN_ID

./venv/bin/python -m suite_tools.scheduler run \
  --contract results/prepared/RUN_ID/sus/RUN_CONTRACT.json \
  --dry-run \
  --max-active-calls 3 \
  --stop-on-attention

./venv/bin/python -m suite_tools.scheduler run \
  --contract results/prepared/RUN_ID/sus/RUN_CONTRACT.json \
  --max-active-calls 3 \
  --stagger-start-seconds 2 \
  --stop-on-attention
```

The scheduler is useful for cockpit visibility and agent operation. It prevents
duplicate launches of the same prepared contract while still allowing runner
parallelism inside a module. Use `suite_tools.prepare_run --module aita` and
`--module epis` for no-paid AITA/Epis contract previews; add
`--output-json --non-interactive` for agent-readable command generation. Add
`--output-json` to scheduler `run`, `stop`, `clear-control`, or `status` when
an agent needs parseable scheduler/control state.

AITA prepared/runtime contracts include an `aita-dataset-manifest-v1` manifest.
For official `nta-paired` runs, verify source CSV hashes, selected pair IDs,
`flip_source=official_aita_nta_flip`, and excluded malformed official rows
before paid collection. Preserve the official CSVs; use manifest exclusions or
a separately versioned cleaned derivative rather than editing them in place.

`RUN_CONTRACT.json` should be written before paid calls begin. It is the
operator-facing answer to "what was supposed to happen?" and lets the dashboard
flag missing artifacts, incomplete expected units, model-id mismatches, and
active stop/pause intent even before scoring. Contract/provenance/dashboard
changes do not change benchmark comparability by themselves. Use
`comparison_spec_hash` to group comparable model runs. It should stay fixed
when only the model-under-test, served endpoint condition, run date, or output
directory changes; it should change when prompts, rubrics, scoring dimensions,
dataset/sample selection, or judge-panel metadata changes. Use
`model_conditions_hash` for the tested model or endpoint condition and
`run_execution_hash` for run-specific audit metadata.

`RUN_CONTROL.json` is cooperative. A request such as
`stop_before_next_paid_call` means the operator or integrated runner should
finish any in-flight provider request, then halt before spending again. It is
not a process manager and must not be treated as arbitrary shell execution.

Treat `RUN_STATUS.json` as the first file to inspect before trusting a run.
Only `status=completed` with `validity=score_ready` should be promoted as
benchmark evidence. Any `failed_*` status means the directory is diagnostic:
read the transcripts if useful, but do not quote the scores as model behavior.

Failure classes are intentionally blunt:

- `failed_auth` or `failed_billing`: stop the paid run; fix the key or credits.
- `failed_invalid`: adapter/backend output was not benchmarkable.
- `failed_incomplete`: one or more conversations did not reach planned turns.
- `failed_scoring`: judge output had missing dimensions or skipped transcripts.
- `failed_provider`: provider/backend transport failure after allowed retries.

The visual review viewer reads `RUN_STATUS.json` and nearest
`RUN_CONTRACT.json` files. It marks records from failed runs as infrastructure
artifacts and embeds contract provenance metadata with each record. This keeps
malformed or partial data visible for debugging without letting it masquerade as
production benchmark evidence.

## Tiny Calibration Commands

These are deliberately small. They are for checking test design, seeker pressure, saving, scoring, and report generation.

### SUS

```bash
cd /path/to/benchmark/sus-bench
python3 -m sus_bench run \
  --models /tmp/benchmark-configs/calibration/sus-models.yaml \
  --runs 1 \
  --scenarios bridge_heights \
  --no-parallel \
  --output results/calibration-sus
```

Re-score saved SUS conversations with a stronger judge panel without re-running the model:

```bash
python3 -m sus_bench rescore \
  results/calibration-sus/sus-bench-YYYYMMDD-HHMMSS-conversations.json \
  --models /tmp/benchmark-configs/frontier/sus-models.yaml \
  --output results/calibration-sus/sus-rescore-frontier.json
```

### AITA

```bash
cd /path/to/benchmark/aita-bench
python3 -m aita_bench run \
  --config /tmp/benchmark-configs/calibration/aita-models.yaml \
  --models gemini-flash \
  --items 1 \
  --allow-sample-fallback \
  --output ../results/aita/aita-calibration-smoke

python3 -m aita_bench score \
  --input ../results/aita/aita-calibration-smoke \
  --config /tmp/benchmark-configs/calibration/aita-models.yaml

python3 -m aita_bench report \
  --input ../results/aita/aita-calibration-smoke \
  --config /tmp/benchmark-configs/calibration/aita-models.yaml
```

`--allow-sample-fallback` is for smoke only. Paid AITA collection should use the
flagship curated set (`data/curated/aita_reversed_n20_v1/`, shipped) via
`nta-paired`, or, for the advanced `yta-synthflip` mode, pass an explicit
full-data `--data` path to a private corpus kept outside the public repo.

### Epistemic

Use the tiny calibration selection to exercise all three adaptive types with only five conversations total.

```bash
cd /path/to/benchmark/epistemic-sycophancy-bench
python3 -m epis_bench run \
  --config /tmp/benchmark-configs/calibration/epis-models.yaml \
  --models gemini-flash \
  --selection data/calibration-selection.yaml \
  --output results/epis-calibration-smoke

python3 -m epis_bench score \
  --input results/epis-calibration-smoke \
  --config /tmp/benchmark-configs/calibration/epis-models.yaml

python3 -m epis_bench report \
  --input results/epis-calibration-smoke \
  --config /tmp/benchmark-configs/calibration/epis-models.yaml
```

Use `python3 -m epis_bench score --force ...` to re-score an existing directory with a different judge.

### Judge Panel Comparison

AITA, Epistemic, and SUS score with the configured judge panel in one score
command. The prepared contract and score artifacts record judge model configs,
rubric source metadata, and judge prompt hashes. AITA and Epistemic artifacts
also preserve each judge's raw score under `judge_scores` while exposing
top-level mean/pass-rate diagnostics and explicit panel-majority release
fields.

Use `--judge-model` only for a deliberate one-off calibration override. Normal
runs should use the rendered `judge_set` in the config so `RUN_CONTRACT.json`
and score artifacts share the same judge panel identity.

`suite_tools.panel_compare` is no longer the production aggregation path. Keep
it for explicit research comparisons of archived/internal historical score
directories, not for current release scoring.

```bash
cd /path/to/benchmark
python3 -m suite_tools.panel_compare \
  results/frontier-judge-smoke-YYYYMMDD \
  --output-dir results/frontier-judge-smoke-YYYYMMDD
```

This writes `panel_comparison.json` and `PANEL_REPORT.md` for archived
comparison work. Treat disagreement rows as calibration signals for
prompt/rubric review, not as human override slots.

## Manual Acceptance Checklist

For each tiny run, inspect transcripts before scaling up:

- The seeker should pressure the model without introducing unrelated facts.
- The model-under-test responses should be saved after every turn.
- `RUN_STATUS.json` should end with `status=completed` and `validity=score_ready`.
- `RUN_EVENTS.jsonl` should show turn/score progress and no terminal failure event.
- The score files should record the judge panel, judge configs, judge prompt
  hashes, rubric metadata, and per-judge scores where applicable.
- Private served systems: ensure your backend's persistence/storage layers are
  disabled for benchmark traffic per your own system's configuration; benchmark
  artifacts must not create user-visible data in the system under test.
- Judge prompts should remain blinded to target model metadata; target model
  identifiers in transcript text should be replaced with `MODEL` before
  scoring.
- Reports should include the expected model key and non-empty scores.
- SUS should expose phase-vs-judge classification conflicts.
- AITA should show both sides of the same conflict when side B exists.
- Epistemic pickside and mirror should preserve the intended opposing side, not accidentally push the same stance twice.
- The visual review viewer should open and show transcript, source path, judge model, and paired score details for every sampled artifact.
- If a transcript feels artificial, fix the prompt/scenario before increasing N.

## Current Known Limits

- SUS rescoring still costs money because it runs judge models over full transcripts.
- Archived historical score directories may be useful for comparison, but the
  current benchmark code assumes the current panel/artifact schema.
- Keep local/private operational notes outside tracked docs; this runbook should
  remain safe for public calibration smokes.
