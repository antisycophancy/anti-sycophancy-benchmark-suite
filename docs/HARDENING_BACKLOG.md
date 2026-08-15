# Benchmark Hardening Backlog

Methodology and robustness work to make the suite defensible as a public
release. Engineering (contracts, provenance, ledgers, gates) is solid; the
open gaps are mostly measurement-science. Grouped by claim-risk — the higher
groups are what a skeptical ML reviewer or a competitor will attack first.

Status legend: ⬜ open · 🔆 in progress · ✅ done · 💤 deferred (deliberate)

Last reviewed: 2026-08-02.

---

## Provider-call dispatch proof remains transport-limited

The local `CALL_DIAGNOSTICS.jsonl` journal records durable intent,
SDK-invocation start, closure, retry identity, safe provider error fields, and
response-shape failures without changing request/response or benchmark
artifacts. The generic OpenAI-compatible SDK boundary still cannot prove the
instant a socket write completed. Unclosed invocations are therefore correctly
reported as billing-ambiguous rather than dispatched. Exact dispatch proof
would require an opt-in transport hook and byte-equivalent-request tests before
it could replace this conservative state.

---

## A. Statistical validity (most exposed)

- ⬜ **A1 — Judge-vs-human validation set.** Every headline number rests on
  LLM-as-judge with no human anchor and no inter-judge agreement reported.
  Build a small human-labeled gold set (50–100 items/module), report
  judge↔human agreement and panel Krippendorff's α / Cohen's κ. *Highest-
  leverage credibility item.* (Open question: is there existing human-labeled
  sycophancy data, or label from scratch?)

- ✅ **A2 — Judge self-preference visibility.** *Done 2026-06-10.*
  `suite_tools/judge_breakdown.py`: per-judge means + paired deltas (judge
  minus other judges on the same items) per model/dimension, grouped per
  module (same-named dims never pooled across modules), same-family
  judge/model rows flagged, ≥0.25 same-family deltas surfaced for review.
  Reads current panel score files (`judge_scores`, aita/epis `judge_model` or
  sus `judge` keys) plus aita's single-judge top-level layout. CLI writes
  `judge_breakdown.json` + `JUDGE_BREAKDOWN.md`; RUNBOOK §0.5 documents it.
  Context: model-under-test identity is ALREADY blinded (`_blind_text` /
  `assert_blind_model_payload`); this makes residual *latent* self-preference
  (Panickssery et al. 2024) visible. Heavier "cross-family-only judge mode"
  remains optional/deprioritized.

- 💤 **A3 — Multi-scenario SUS.** All SUS comparison data uses one scenario
  (`bridge_heights`) — n=1 on the construct axis. Additional scenarios exist
  privately; deliberately NOT releasing/burning them yet. Revisit when ready
  to publish a versioned scenario expansion. (Held in `private_question_bank/`.)

- ⬜ **A4 — Multiple-comparison discipline.** N=20 × 5–6 dimensions × 6+ models
  with no correction; leaderboard "A beats B" claims with overlapping CIs.
  Report effect sizes via the existing pairwise sign test, mark non-significant
  diffs explicitly, state minimum detectable effect at N=20. (CI t-table bug
  already fixed 2026-06-10.)

- ⬜ **A5 — Pin and report sampling temperature.** Stochastic generation at
  unstated temperature → not replicable; sampling noise conflated with item
  variance. Pin temperature into the condition (already affects
  `model_conditions_hash` when set), report it, optionally k-samples/item to
  separate sampling from item variance.

## B. Construct & adversarial validity

- ✅ **B6 — Judge prompt-injection hardening.** *Done 2026-06-10.*
  `suite_tools/prompt_fencing.py`: `fence()` delimiter-wraps model-under-test
  text after neutralizing any closing tag the model emits (zero-width break →
  can't escape the fence); reusable data-not-instructions guard. Applied to
  all SUS analyzer/judge prompts + bridge.yaml scenario, AITA + Epis
  JUDGE_FRAMEWORK and every judge format site. Regex tightening: verdict
  acronyms standalone-uppercase only, dropped bad `\binfo\b`, NAH→NTA,
  `_extract_score` requires labeled format before loose fallback.
  Injection-resistance tests per module (fail pre-fix). Bumped
  benchmark_spec_hash (pre-publication, intended).

- ⬜ **B7 — Seeker/analyzer quality validation.** The adaptive seeker generates
  the pressure; a weak seeker makes every model look resistant. Add a seeker-
  quality check (does it actually escalate?) and a seeker-model sensitivity
  analysis (do rankings hold under a different seeker model?).

- ⬜ **B8 — AITA pairing-integrity test.** Both-sides consistency depends on
  NTA/YTA pairing integrity. Add a dedicated test; over-broad verdict regexes
  (`\bnah\b`, `\binfo\b`) and the yta-synthflip contract ordering were partial
  offenders (some already fixed).

- ⬜ **B9 — Freeze synthetic flips before tested-model spend.** The prepared
  `yta-synthflip` contract freezes the source items and flip generator but not
  the generated side-B text. A fixed cleared paired dataset is the safest v1
  publication route. A future stochastic mode needs a two-stage,
  append-only materialization contract whose flip hashes are authenticated
  before any model-under-test call.

## C. Reproducibility & drift

- ⬜ **C9 — Model drift documentation + fingerprint capture.** `*-preview`
  models are moving targets; the hash captures the slug, not the weights.
  Document prominently; capture provider `system_fingerprint`/version into the
  condition where available.

- ⬜ **C10 — Frozen reference run.** Ship one small frozen result bundle +
  expected scores as an integration fixture so a fresh clone can confirm its
  setup reproduces known numbers.

## D. Code/architecture cleanup (lower claim-risk)

- ✅ **D17 — Resume condition-identity closure.** *Done 2026-08-02.* SUS now
  persists rendered condition metadata in every unit transcript; SUS, AITA, and
  Epistemic validate saved identity before reuse; SUS scoring fails before judge
  calls on missing or conflicting identity; `bench verify` and evidence
  packaging fail closed on checkable transcript/contract mismatches. A
  no-provider derived materializer records unambiguous source-contract identity
  restoration without editing source artifacts. Incident record:
  `docs/incidents/2026-08-02-sus-resume-condition-identity.md`.

- ✅ **D11 — Fail-closed gaps.** *Done 2026-06-10.* epis `benchmark_family_id`
  unified to `epistemic` (matches module key; explicit identity and fallback
  derivation can no longer fork the spec hash — spec-hash change,
  pre-publication, intended); epis judge-panel failure now continues scoring
  remaining items (run still fails closed at the end); paid-call leases record
  `host` and are reclaimed immediately when the local holder PID is dead
  (foreign-host leases keep the 30-min ceiling); scheduler lock takeover is
  rename-claim atomic — a challenger holding a stale read can no longer unlink
  a fresh lock (double-spend window closed). All TDD'd.
- ✅ **D12 — Source packaging.** *Done 2026-08-14.* The root meta-package,
  hash-locked binary dependency set, authenticated-source bootstrap, clean
  export, and fresh-install tests support the public source distribution on
  Python 3.11-3.13. Standalone wheel distribution remains outside v1 scope.
- ✅ **D13 — aita-bench README.** *Done 2026-06-10.* Rewritten around the real
  surface: verdict-alignment primary dims (+ `*_majority` panel fields),
  diagnostic/mechanism split per the scoring contract, contract-first suite
  workflow, both dataset modes, real results layout. Fictional BaseScorer /
  PyPI / `scenarios/` sections removed. Stale pre-v3 `results/sample`
  artifacts deleted (they misrepresented current output); a fresh frozen
  sample bundle is folded into **C10** rather than regenerated ad hoc.
- ✅ **D14 — Epis data pinning.** *Done 2026-06-10.* Syco-bench CSVs vendored
  verbatim (MIT-0, upstream commit `5ecc005b`) into
  `epistemic-sycophancy-bench/data/syco_bench/` with LICENSE + PROVENANCE.md;
  tests pin sha256 per file and require the in-repo path. No external clone
  in setup; item content byte-identical so sample_spec hashes unchanged.

## E. Coverage (scope expansion, not bugs)

- ⬜ **E15 — Over-refusal / helpfulness retention axis.** A model that refuses
  everything scores great on SUS but is useless. Add a paired helpfulness-
  retention measure so SUS can't be gamed by becoming uselessly cautious.
- ⬜ **E16 — Breadth:** English-only, US-centric scenarios; small fixed item
  banks; no difficulty stratification. Expand deliberately, versioned.

---

## Recommended order for a defensible public release

1. ~~**B6** (injection hardening)~~ ✅ done 2026-06-10
2. ~~**A2** (per-judge breakdowns)~~ ✅ done 2026-06-10
3. **A4** (significance discipline on the leaderboard)
4. **A1** (human gold set — biggest lift, biggest credibility payoff)
5. ~~D-group cleanup~~ ✅ D11-D14 done; standalone wheels remain outside v1

A3 (more SUS scenarios) is intentionally deferred to avoid burning the private
question bank before publication.
