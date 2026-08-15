# Benchmark Operator Commands

Use these recipes from the benchmark repository root. Replace uppercase
placeholders deliberately. Run `--help` before using a command when the local
checkout may have changed.

## Verified Offline Setup

Start only from an exact cryptographically signed release tag or a release
archive whose detached signature you verified against the independently
announced release signer. The archive's embedded `SHA256SUMS` proves integrity
and inventory only; it does not authenticate the publisher.

```bash
./scripts/verify-release-source
PYTHON_BIN=python3 ./scripts/bootstrap
test -e .env || (umask 077 && cp .env.example .env)
chmod 600 .env

./venv/bin/python -m suite_tools.model_config --validate
./venv/bin/python -m suite_tools.model_config --list --output-json
```

Bootstrap installs the frozen, hashed, binary-only dependency lock, installs
all four local packages without dependency resolution, and runs the full
offline gate. It refuses CPython versions outside 3.11 through 3.13 and will
not merge into an existing `venv`.

Do not treat an offline-gate skip as acceptable merely because a previous
release had one. Read the printed reason and accept only a currently documented
optional/private test.

## Existing OpenAI-Compatible Endpoint

Put private URLs, credential names, and internal route labels in an ignored
registry overlay where needed. Model keys, model IDs, and display labels enter
public evidence, so give them publication-safe aliases before preparing a run
that may be packaged. The minimum registry shape is:

```yaml
extends: ../suite_models.yaml

endpoints:
  my_endpoint:
    provider_api: openai_compatible
    openai_base_url: "https://example.test/v1"
    chat_completions_url: "https://example.test/v1/chat/completions"
    api_key_env: MY_ENDPOINT_API_KEY

models:
  my-model:
    model_id: "organization/model-id"
    label: "My Model"
    endpoint: my_endpoint

model_groups:
  my_model_smoke:
    - my-model
```

Custom key variables default to loopback-only. For an intentional remote HTTPS
endpoint, authorize the exact hostname separately from the registry:

```bash
export BENCHMARK_ALLOWED_ENDPOINT_HOSTS=example.test
```

Official provider key variables cannot be redirected, and a missing custom key
never falls back to the OpenRouter key.

Set `MY_ENDPOINT_API_KEY` in `.env`, not YAML or a run contract. Then validate
the overlay and probe the exact condition:

```bash
./venv/bin/python -m suite_tools.model_config \
  --config private_profiles/my-suite-models.yaml --validate

./venv/bin/python -m suite_tools.preflight_conditions \
  --suite-config private_profiles/my-suite-models.yaml \
  --group my_model_smoke
```

Use the same overlay when preparing the benchmark contract. Otherwise the
prepared contract will use the public registry instead of the endpoint just
proved:

```bash
./venv/bin/python -m suite_tools.prepare_run \
  --module sus \
  --run-id my-endpoint-smoke \
  --config private_profiles/my-suite-models.yaml \
  --models group:my_model_smoke \
  --judge-set calibration \
  --scenarios bridge_heights \
  --runs 1 \
  --output results/prepared/my-endpoint-smoke \
  --non-interactive --output-json
```

If using the bundled reference adapter, start it in reference mode and run its
contract smoke first:

```bash
./venv/bin/python adapter/server.py
./venv/bin/python adapter/smoke.py
```

Remote proxy mode can make a paid call and requires explicit consent:

```bash
./venv/bin/python adapter/smoke.py \
  --base-url http://127.0.0.1:9999/v1 \
  --api-key-env LOCAL_OPENAI_COMPATIBLE_API_KEY \
  --allow-proxy-call
```

This smoke is for the bundled adapter, which provides `/health` and
`/v1/models`. Do not require those extra routes from an unrelated existing
OpenAI-compatible endpoint. For that case, use exact-condition preflight; the
benchmark contract itself requires `/v1/chat/completions`.

See `adapter/README.md` before changing the adapter.

## Prepare And Dry-Run One Public-Registry Contract

The beginner default uses the public `suite_models.yaml` and its current
calibration smoke group. List the registry first and explain the resolved model
and provider before preparing it.

SUS smoke:

```bash
./venv/bin/python -m suite_tools.prepare_run \
  --module sus \
  --run-id RUN_ID \
  --models group:calibration_smoke \
  --judge-set calibration \
  --scenarios bridge_heights \
  --runs 1 \
  --output results/prepared/RUN_ID \
  --non-interactive --output-json
```

AITA calibration smoke:

```bash
./venv/bin/python -m suite_tools.prepare_run \
  --module aita \
  --run-id RUN_ID \
  --models group:calibration_smoke \
  --judge-set calibration \
  --items 1 \
  --dataset-mode yta-synthflip \
  --allow-sample-fallback \
  --output results/prepared/RUN_ID \
  --non-interactive --output-json
```

This AITA command proves the lifecycle with three clearly synthetic bundled
rows; it is not a publication condition. Publication work uses the separately
signed sealed `nta-paired` N=20 pack identified by
`manifests/aita-sealed-pack-v1.json`. Follow `RUNBOOK.md` and pass
`--sealed-pack`; the authenticated locked selection supplies all 20 items.

Inspect availability without networking. Fetch only after the user has seen
the exact receipt and approved the external download.

```bash
./venv/bin/python -m suite_tools.aita_data_pack status --json

./venv/bin/python -m suite_tools.aita_data_pack fetch \
  --destination private_question_bank/aita-reversed-n20-v1 \
  --confirm-download \
  --json
```

For Epistemic, keep the three types visible:

```bash
./venv/bin/python -m suite_tools.prepare_run \
  --module epis \
  --run-id RUN_ID \
  --models group:calibration_smoke \
  --judge-set calibration \
  --items 1 \
  --types delusion,pickside,mirror \
  --selection epistemic-sycophancy-bench/data/calibration-selection.yaml \
  --output results/prepared/RUN_ID \
  --non-interactive --output-json
```

After preparation, preflight the frozen module condition and dry-run its
scheduler command:

```bash
./venv/bin/python -m suite_tools.preflight_conditions \
  --run-dir results/prepared/RUN_ID/sus

./venv/bin/python -m suite_tools.scheduler run \
  --contract results/prepared/RUN_ID/sus/RUN_CONTRACT.json \
  --run-pace cautious --stop-on-attention --dry-run --output-json
```

Every preflight target is a network request. The bundled local reference
adapter is free; remote or proxy endpoints may bill.

## Observe And Run One Contract

```bash
./venv/bin/python -m suite_tools.live_dashboard \
  --results-root results/prepared --port 8766
```

In another terminal, after paid-run approval:

```bash
./venv/bin/python -m suite_tools.scheduler run \
  --contract results/prepared/RUN_ID/sus/RUN_CONTRACT.json \
  --run-pace cautious --stop-on-attention
```

When generation is clean and parked at Needs Scoring:

```bash
./venv/bin/python -m suite_tools.scheduler score \
  --contract results/prepared/RUN_ID/sus/RUN_CONTRACT.json \
  --stop-on-attention
```

## Run Independent Contracts Together

Prepare each child independently, then preflight every frozen run directory:

```bash
./venv/bin/python -m suite_tools.preflight_conditions \
  --run-dir results/prepared/RUN_ONE/sus \
  --run-dir results/prepared/RUN_TWO/aita \
  --run-dir results/prepared/RUN_THREE/epis
```

Dry-run the fleet before spend:

```bash
./venv/bin/python -m suite_tools.scheduler run-many \
  --contract results/prepared/RUN_ONE/sus/RUN_CONTRACT.json \
  --contract results/prepared/RUN_TWO/aita/RUN_CONTRACT.json \
  --contract results/prepared/RUN_THREE/epis/RUN_CONTRACT.json \
  --run-pace normal \
  --max-active-calls N \
  --stop-on-attention \
  --dry-run --output-json
```

Remove `--dry-run` only after approval. All children share the global paid-call
lease. `--stop-on-attention` stops an affected child, not its siblings. Clean
children auto-score by default; add `--no-auto-score` to keep a manual scoring
gate.

Do not replace `run-many` with shell background jobs. It would lose the shared
lease and the scheduler's contract-level visibility.

## Inspect Or Resume

```bash
./venv/bin/python -m suite_tools.bench status results/prepared/RUN_ID/sus
./venv/bin/python -m suite_tools.bench diagnose results/prepared/RUN_ID/sus --json
./venv/bin/python -m suite_tools.bench verify results/prepared/RUN_ID/sus
./venv/bin/python -m suite_tools.hygiene_gate results/prepared/RUN_ID/sus
./venv/bin/python -m suite_tools.owed_units results/prepared/RUN_ID/sus
```

After fixing the actual environment failure, clear a stale cooperative stop and
rerun the same frozen contract:

```bash
./venv/bin/python -m suite_tools.scheduler clear-control \
  --contract results/prepared/RUN_ID/sus/RUN_CONTRACT.json

./venv/bin/python -m suite_tools.scheduler run \
  --contract results/prepared/RUN_ID/sus/RUN_CONTRACT.json \
  --run-pace cautious
```

Completed units are reused according to module rules only after their saved
`condition_id` and `condition_hash` agree with the same rendered condition. Do
not move ledgers or edit the contract to force pickup. Before scoring, require:

```bash
./venv/bin/python -m suite_tools.bench verify \
  results/prepared/RUN_ID/MODULE --json
```

Both `request_conformance.conformant` and
`artifact_identity.conformant` must be true.

The hygiene command is offline and observation-only. It exits non-zero when
saved transcripts contain blocking runtime-error text, empty responses,
malformed wrappers, or incomplete conversations; it never edits run artifacts
or changes contract/comparison hashes.

## Materialize An Offline Publication Subset

Use this only when the AITA or Epistemic source is already completed and
`score_ready`. It makes no provider calls and records an immutable exclusion:

```bash
./venv/bin/python -m suite_tools.materialize_subset \
  --source-run-dir results/prepared/SOURCE_RUN/aita \
  --output-dir results/prepared/DERIVED_RUN/aita \
  --run-id DERIVED_RUN \
  --exclude-model BAD_MODEL_KEY_ONE \
  --exclude-model BAD_MODEL_KEY_TWO \
  --reason "Documented publication exclusion"
```

The tool copies selected artifacts byte-for-byte and pins their source hashes.
Repeat `--exclude-model` for every affected contract model key. Use
`bench verify --json` to identify unverifiable condition identities, then map
each identity to its model key in `RUN_CONTRACT.json`. The derived run can
preserve unaffected conditions from a mixed pre-receipt composite, but it can
never retain or validate an affected condition and cannot rescue incomplete
generation. For a saved SUS rescore, merge, or serialization defect whose
missing identity maps unambiguously to one frozen source-contract condition,
inspect `suite_tools.materialize_sus_derived --help`. It makes no provider
calls, writes a new derived run, and records source hashes and restored fields;
it must not be used as an in-place repair or to override a conflict.

## Evidence Package

Initialize an experiment once, then use the unified bench CLI for adoption and
evidence packaging. For a resumable packaging workflow, use the dedicated goal:

```bash
./venv/bin/python -m suite_tools.companion start WORKFLOW \
  --goal evidence_package --json

./venv/bin/python -m suite_tools.experiment init \
  results/experiments/EXPERIMENT_ID \
  --id EXPERIMENT_ID --title "Experiment title" \
  --from-run results/prepared/RUN_ID/sus

./venv/bin/python -m suite_tools.bench adopt \
  results/experiments/EXPERIMENT_ID \
  results/prepared/ANOTHER_RUN/aita --role primary

./venv/bin/python -m suite_tools.bench review --json
./venv/bin/python -m suite_tools.bench package \
  results/experiments/EXPERIMENT_ID --out results/bundles --json
./venv/bin/python -m suite_tools.bench verify \
  --bundle results/bundles/BUNDLE_DIR --json
```

Read `RUNBOOK.md` sections 0.7 and 0.8 before dispositioning evidence. A review
decision is a scientific record, not a cleanup action.

## CLI Map

| Need | Command |
| --- | --- |
| Registry validation and listing | `./venv/bin/python -m suite_tools.model_config` |
| No-paid contract creation | `./venv/bin/python -m suite_tools.prepare_run` |
| Exact endpoint/effort probe | `./venv/bin/python -m suite_tools.preflight_conditions` |
| One or many scheduled contracts | `./venv/bin/python -m suite_tools.scheduler` |
| Owed-unit and provenance review | `./venv/bin/python -m suite_tools.bench status/verify` |
| Evidence disposition | `./venv/bin/python -m suite_tools.bench review` |
| Publication bundle | `./venv/bin/python -m suite_tools.bench package` |
| Adapter contract test | `./venv/bin/python adapter/smoke.py` |
