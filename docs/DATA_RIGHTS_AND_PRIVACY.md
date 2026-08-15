# Data Rights and Privacy

This repository combines original benchmark software with source material from
third parties. Those are different release surfaces and have different rights.

## Code and data are licensed separately

The repository's MIT license covers the benchmark software and documentation
created for this project. It does not grant rights in third-party source
content. The public software tree contains no Reddit-derived prompt text,
source URLs, post IDs, labels, or locked selection. The runnable AITA N=20
condition is a separately signed sealed data pack. Source authors and platforms
retain whatever rights they hold in that material; pack hashes and attribution
are provenance records, not a new license for the source text.

The project-created reversed perspectives and benchmark annotations are
distributed for research use in that separate pack, but they do not expand the
rights available in the underlying source posts. Users are responsible for confirming that
their collection, redistribution, processing, and publication of benchmark
inputs or outputs is lawful and consistent with applicable platform terms.

The optional ELEPHANT-derived modes use data obtained separately from the
upstream CC0 release. The full upstream CSV files are not distributed here.

## Personal information

Public social-media text can contain names, usernames, locations, health or
family details, and other personal information even when it was posted openly.
Do not use this dataset to identify, contact, profile, or make decisions about
the people described in it. Do not combine it with other data to re-identify
authors or subjects.

Before a dataset version is published, maintainers should review every included
item and synthetic reversal for direct identifiers that are not needed for the
benchmark. Email addresses, phone numbers, street addresses, account handles,
and similar identifiers should be removed or replaced while preserving the
conflict needed for evaluation. Local run artifacts can reproduce source text.
Public bundle export omits conversation text by default, and sealed-pack runs
cannot opt into raw transcript publication; operators must still review every
derived artifact before sharing.

This process reduces exposure; it is not a guarantee that all sensitive or
identifying context has been removed.

## Where run data goes

Running the suite sends data to third parties twice, not once. The obvious leg
is generation: benchmark items go to the provider hosting the model under test.
The less obvious leg is **scoring** — judges receive the model-under-test
transcripts, and the default judge sets route through OpenRouter to OpenAI,
Anthropic, and Google upstreams, so those outputs land in third-party
infrastructure under those providers' retention terms. The adaptive seeker and
analyzer see transcripts too. Target identity is blinded before judging; the
content is not.

Operators evaluating a private or unreleased system should treat this as an
egress decision, not a detail. Judges can be repointed at your own endpoint via
`endpoints:` / `judge_models:` / `judge_sets:` in `suite_models.yaml` (see the
root README, "Before you point this at a real key"). Beyond provider calls the
harness has no analytics, telemetry, or webhook path, and the dashboard binds
`127.0.0.1` by default.

## Corrections and removal

Send correction or removal requests to `research@antisycophancy.ai`. Include
only the Reddit post ID or benchmark item ID and request a private follow-up;
do not send personal or sensitive information through the public issue tracker.
Maintainers should remove the item from future dataset versions and document
the change; released, hash-bound versions must not be silently edited in place.

## Release checklist

The current machine-readable software decision is
`manifests/aita-data-clearance.json`: `status=cleared`, recorded by the human
release owner on 2026-08-14. It records that Reddit-derived plaintext is
excluded from the tracked source release. `manifests/aita-sealed-pack-v1.json`
separately records the external pack's immutable hashes, byte-parity migration
evidence, split-key locations, and removal contact. Neither record relicenses
third-party source text or substitutes for a downstream user's own obligations.

`scripts/export-release` validates that record against every exported AITA data
path and fails closed if a path is uncovered or the decision fields become
incomplete. A later correction or removal produces a new version or tombstone;
hash-bound published pack bytes are never silently rewritten.

For each public dataset version, maintainers must record:

- the source and redistribution basis;
- the direct-identifier review date and reviewer;
- any exclusions, redactions, corrections, or removal requests;
- the content hashes and versioned manifest used by benchmark runs.

If the redistribution basis or identifier review is not documented, the
dataset is not cleared for a public release even if the software tests pass.
