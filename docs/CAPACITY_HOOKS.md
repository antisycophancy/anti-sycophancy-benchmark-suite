# Pre-Run Capacity Intent Hooks

Some benchmarks call a private or self-hosted model endpoint. That endpoint may
need to scale workers, warm caches, raise queue limits, or reserve capacity
before the run starts. The benchmark suite exposes this need through a generic,
side-effect-free intent file:

```text
RUN_CONTRACT.json -> CAPACITY_INTENT.json -> operator/private infrastructure hook
```

The public benchmark code only creates `CAPACITY_INTENT.json`. It does not call
Render, Kubernetes, ECS, a cloud autoscaler, a database, or the model endpoint.
Private operators can consume the intent with their own wrapper.

## Create an intent

```bash
./venv/bin/python -m suite_tools.capacity_intent \
  --contract results/prepared/RUN_ID/sus/RUN_CONTRACT.json \
  --match-model-prefix private-endpoint/ \
  --provider-calls-per-turn 3 \
  --calls-per-capacity-unit 10 \
  --json
```

By default the intent is written beside the contract as `CAPACITY_INTENT.json`.
If no match rules are supplied, all expected model units in the contract are
treated as matching. Use match rules when only one endpoint family needs a
capacity signal.

Supported public match rules:

- `model_id_prefixes`: match model ids by prefix.
- `endpoint_names`: match exact endpoint names from `suite_models.yaml`.
- `endpoint_contains`: match endpoint names by substring.

Supported public sizing fields:

- `default_turns_per_unit`: fallback turn count when a unit has no planned
  turn count.
- `provider_calls_per_turn`: expected private/provider calls per benchmark
  turn.
- `default_max_active_calls`: upper bound for scheduler concurrency when not
  overridden.
- `calls_per_capacity_unit`: how many active calls one private capacity unit is
  expected to carry.
- `min_capacity_units` / `max_capacity_units`: advisory bounds for the target
  capacity.

The output includes:

- `estimate.total_units`, `estimate.matching_units`, `estimate.planned_turns`,
  and `estimate.estimated_provider_calls`.
- `capacity.max_active_calls` and `capacity.target_capacity_units`.
- `estimate.matching_model_conditions`, including opaque model-condition fields
  such as `condition_hash`, `served_profile_hash`, and
  `provider_condition_hash` when the prepared contract contains them.
- `side_effects: "none"` and a `contract_invariance` block stating that the
  intent does not modify prompts, questions, model payloads, judges, scoring,
  artifacts, or the run contract.

## Private hook pattern

A private capacity wrapper should:

1. Prepare a normal benchmark `RUN_CONTRACT.json`.
2. Generate or read `CAPACITY_INTENT.json`.
3. Translate `capacity.target_capacity_units` into its own infrastructure
   vocabulary, such as worker count, instances, pods, or queue slots.
4. Optionally wait until the service is healthy.
5. Launch the normal scheduler command against the unchanged contract.
6. Optionally restore baseline capacity after the run.

The private hook may be Render-specific, Kubernetes-specific, or entirely local.
That implementation should stay outside the public benchmark surface unless it
is generally useful and contains no private service details.

## Provenance

Capacity intent files are operational metadata, not benchmark results. They are
safe to keep with run diagnostics because they carry only model ids, endpoint
names, and opaque condition hashes from the prepared contract. They must not
include raw private prompts, private backend configuration names, secrets, API
keys, authorization headers, or full private request payloads.

Changing capacity does not change scientific comparability by itself. The
normal provenance keys remain in `RUN_CONTRACT.json`: benchmark spec hash,
sample hash, judge panel hash, model condition hashes, and run execution hash.

## Strict per-call ceilings

`PaidCallLeaseManager.acquire(max_active_calls=N)` compares `N` with the total
number of active leases across all callers. It does not reserve `N` slots for
that waiter. Under sustained traffic admitted by a higher shared policy, a
strict-cap waiter can therefore wait until its lease timeout expires. This is
intentional after the work-conserving convoy fix: the override exists for
probes and diagnostics, not normal benchmark scheduling. Runners derive worker
counts from the shared capacity policy through
`effective_paid_call_parallelism`.
