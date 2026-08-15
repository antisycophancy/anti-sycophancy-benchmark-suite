# V1 Public Release Checklist

This checklist is the release decision record for the benchmark suite, public
dashboard, reference adapter, command-line tools, and `/antisycophancy` agent
skill. A green technical check is evidence about the tested source, not
permission to publish data or create public accounts.

## Release status

This checklist is not authorization to publish. The source exporter remains
fail-closed until the data-rights record permits every included AITA source
file, and maintainers must close every human gate below for the exact artifact
they intend to sign and upload.

## Automated technical gates

- [x] Only the root `adapter/` implementation is canonical; the duplicate AITA
  adapter has been removed.
- [x] Non-loopback adapter binds require inbound authentication, request bodies
  are bounded, and the full transcript plus optional correlation id are
  forwarded.
- [x] SUS HTML output escapes untrusted model content and carries a restrictive
  content security policy.
- [x] The live dashboard is loopback-only, rejects hostile Host/peer requests,
  requires process-bound CSRF confirmation, and records operator-attributed
  dispositions in an fsynced append-only ledger before updating snapshots.
- [x] Every benchmark runner validates immutable contract provenance before a
  provider call. Current contracts bind full judge configuration, prompts,
  declared source-item universe, model conditions, routes, and panel identity
  without reblessing old hashes. The separate synthetic-flip gate below limits
  the AITA claim until side-B text is materialized.
- [x] Bundling fails closed on missing or stale provenance, empty members,
  unresolved evidence, non-publishable units, and broken score projections.
- [x] The working release surface contains no private paths, result artifacts,
  secret-shaped values, stale public names, or future repository claims.
- [x] The source exporter requires a clean full Git SHA, reproduces the exact
  tree, rejects unsafe members, writes `SHA256SUMS`, and audits the result.
- [x] Git-free exports verify `SHA256SUMS`; missing, altered, or unexpected
  inventory fails closed. This is corruption/inventory evidence only, not
  publisher authentication; public archives still require a detached signature
  anchored to the independently announced signing identity.
- [x] Fresh Git installs require a clean checkout at an exact cryptographically
  signed tag; bootstrap rejects branches, unsigned tags, tracked drift,
  untracked source, symlinks, and submodules before Python starts.
- [x] Third-party packages are exact-version, hash locked, binary-only, and
  installed before the four local packages with no dependency or build-isolation
  lookup. Python 3.11, 3.12, and 3.13 are the supported release matrix.
- [x] Dependency audit reports no known vulnerability in the fresh Python 3.11
  release environment. Medium/high Bandit scan is clean.
- [x] CI uses immutable action SHAs and runs clean-export bootstrap, all module
  tests, CLI help/version checks, release audit, dependency audit, and static
  security scan across Python 3.11-3.13.
- [x] Every installed benchmark package exposes a working command (`bench`,
  `sus-bench`, `aita-bench`, and `epis-bench`); module entry points and
  `--version`/`--help` paths are exercised in tests and CI.
- [x] `/antisycophancy` and `$antisycophancy` are the canonical skill commands.
  Repository-local Claude Code and Codex discovery wrappers load the same
  canonical guide. Its modes cover connection, run, resume, review, and package
  workflows.
- [x] Metric direction is explicit: AITA resistance is higher-better; SUS
  failure is higher-worse while the legacy safety transform is higher-safer;
  Epistemic contains mixed dimensions.
- [x] Cost guidance computes uncached input, cached input, output, and billed
  reasoning tokens from a dated pricing snapshot. Missing usage or price data is
  `unknown`, never silently zero. SUS ledgers preserve provider-reported versus
  estimated cost provenance through generation and scoring.
- [x] A bounded live SUS smoke completed on 2026-08-13 with one scenario,
  concurrency one, a frozen OpenRouter pricing snapshot, and a $0.30 warning
  ceiling. The successful attempt used Gemini 3 Flash as the model under test
  and Gemini 3.1 Pro as the single judge: 7 calls, 7,101 input tokens, 2,578
  output tokens (including 880 thinking tokens), and $0.0279 provider-reported
  cost. Contract, request, artifact, run, and packaged-bundle verification all
  passed for the completed attempt.

## Human release gates

- [x] **AITA software-tree data clearance.** The human release owner approved
  the 2026-08-14 record that excludes Reddit-derived plaintext, source URLs,
  post IDs, labels, and selection from the tracked software release. The only
  in-tree AITA prompts are clearly marked synthetic smoke fixtures.
- [ ] **AITA sealed-pack release authorization.** Public v1 uses the fixed
  `aita_reversed_n20_v1` condition: 20 originals, 20 project-created reviewed
  reversals, explicit labels, a locked selection, and per-side hashes. Before
  publishing the separate pack, verify its signed release, suite-release Part B
  asset, data-rights decision, correction/removal channel, and exact hashes in
  `manifests/aita-sealed-pack-v1.json`. Prepared
  `yta-synthflip` runs
  cannot claim an exact frozen side-B sample until the generated reversals are
  materialized and hash-bound before tested-model spend, so that legacy mode is
  not the public v1 comparison condition.
- [ ] **Public contact delivery.** `research@antisycophancy.ai` is the declared
  package, citation, correction, and removal contact. Verify Cloudflare email
  routing/DNS, inbound receipt, reply handling, and responsible owners before
  making the release public.
- [ ] **Private credential review.** Authorized maintainers must complete the
  private history and credential review, rotate or revoke anything uncertain,
  and record the decision outside the public source artifact. Never publish or
  mirror the inherited development history.
- [ ] **Clean public history.** Create the public repository from the audited
  exported tree as a new root commit. Do not push, mirror, or merge this
  repository's inherited history.
- [ ] **Final identity.** Confirm release version/tag, organization account,
  repository and issue-tracker URLs, signing identity, and maintainers.
- [ ] **Public artifact inspection.** Review the exact exported `SHA256SUMS`,
  license and citation metadata, README, sample data, dashboard output, and any
  benchmark bundle before signing or uploading.

## Final release procedure

1. Close the separate AITA sealed-pack authorization gate without weakening the
   already-cleared software-tree validator.
2. Commit the candidate privately and run `scripts/export-release` against that
   full immutable SHA and final version into an empty destination.
3. In the Git-free export, run `./scripts/bootstrap`, the installed CLI
   help/version checks, `suite_tools.release_audit --strict`, `pip-audit`, and
   the medium/high Bandit scan on each supported Python version.
4. Compare the export inventory to `SHA256SUMS`, perform the human public
   artifact inspection, and create/verify a detached archive signature against
   the independently announced signing identity. Never present the embedded
   manifest alone as publisher authentication.
5. From the Git-free export, seed an empty public root.

   ```bash
   ./scripts/seed-public-root \
     --out <empty-dir> \
     --release-version <version>
   ```

   This verifies the embedded inventory and copies only manifest-listed source
   files. It deliberately omits the generated `SHA256SUMS`, which is reserved
   for Git-free exports and must never be tracked.
6. Initialize Git in that seeded directory, create the new public root commit,
   sign the final tag, and enable the same CI gates before accepting changes.
   Run `scripts/export-release` from the new root commit and repeat bootstrap,
   audit, and artifact inspection on that second export before publication.
7. If the final exported artifact differs in runtime behavior from the recorded
   candidate smoke, obtain a new explicit spend approval and repeat the bounded
   smoke. Preserve its contract, usage, provider identifiers, and unmodified
   response artifacts outside source.

## Proof boundary

The adapter's `conversation_id` is correlation metadata, not a security or
integrity primitive. Conversation continuity comes from replaying the saved
message transcript. Scientific integrity comes from immutable condition and
artifact hashes, append-only ledgers, pre-spend contract validation, bundle
verification, and a clean exported source checksum manifest.

Prepared-config hashing prevents accidental or post-prepare drift before
spend. It is not a defense against an attacker who can modify the running
process or replace an authenticated file between validation and use; same-host
process compromise and external signing/transparency are outside the v1 local
trust boundary.
