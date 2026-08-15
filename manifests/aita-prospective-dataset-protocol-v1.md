# AITA Prospective Dataset Protocol v1

Status: protocol draft, no collection performed
Created: 2026-05-26
Scope: future AITA/AITA-style paired-perspective dataset collection

## Purpose

This protocol records, before any new collection, how a future AITA dataset
would be built from recently available public advice-conflict posts. The goal is
to avoid post-hoc cherry picking, preserve comparability with ELEPHANT-derived
AITA runs, and create a contamination-aware path for later benchmark expansion.

This protocol does not authorize scraping, API use, redistribution, or model
training. Collection may only proceed after confirming source-platform terms,
API access approval, privacy requirements, storage limits, and publication
rights.

## Relationship To Current AITA Data

Current official AITA paired-consistency runs should use a curated subset of the
official ELEPHANT AITA-NTA paired files, preserving upstream hashes and row IDs.

Future prospective data should be treated as a separate benchmark condition:

- `aita_reversed_n20_v1`: current release set — construct-reversed flips derived
  from a newer set of AITA Reddit conversations (shipped).
- `aita_nta_paired_curated_v1`: retired — curated official ELEPHANT/OSF paired
  flips (superseded by v2, moved to the private tree).
- `aita_prospective_reddit_v1`: newly collected post-release source posts.
- `aita_prospective_flipped_v1`: prospective posts plus generated/reviewed
  opposing-perspective rewrites.

Do not mix these conditions in the same headline score without reporting the
condition boundary.

## Source Access Requirements

Before collection, record:

- source platform and access method
- API application id or approved access class, excluding secrets
- terms/version checked date
- rate limits and backoff policy
- allowed storage/retention policy
- redistribution/publication constraints
- privacy/deletion handling
- whether the data may be used for benchmark evaluation, model training, both,
  or neither

For Reddit-derived data, the intended path is official authenticated API access
or another explicitly permitted research access path. Do not use bypass
scraping, masked user agents, multiple apps for one use case, or non-compliant
archives.

## Candidate Collection Frame

Record the collection frame before fetching content:

- subreddit or source community
- time window
- sort/listing/search method
- maximum candidate count
- inclusion labels sought, such as YTA/NTA consensus
- minimum score/comment thresholds, if any
- language filter
- excluded categories, including deleted/removed/locked posts, minors,
  explicit identifying information, medical/legal crisis content, or posts
  without a clear interpersonal conflict

The frame must be broad enough that the final sample is not handpicked for
expected model behavior.

## Candidate Fields

Store only fields needed for reproducibility and audit:

- platform post id and stable URL
- created timestamp
- title
- body text
- source label or community verdict
- top comment id and text when permitted
- score/upvote/comment metadata when permitted
- retrieval timestamp
- raw record hash
- normalized prompt hash
- redaction notes

Do not store usernames unless a future compliance review concludes they are
necessary and permitted.

## Perspective Flip Generation

If opposing-perspective rewrites are generated, record:

- source post hash
- generator model id
- provider/endpoint class
- prompt/rubric version
- decoding parameters
- generation timestamp
- output hash
- failure reason for non-rewritable posts

Generated flips must be reviewed before use. They are not replacements for
official ELEPHANT flips and must remain a separate condition.

## Flip Quality Criteria

A candidate flip is acceptable only if it:

- rewrites the same conflict from the opposing party's first-person perspective
- preserves material facts without adding new evidence
- remains coherent as an AITA-style advice-seeking post
- avoids answering the moral question in the prompt itself
- avoids explicit benchmark, prompt, or model artifacts
- is long enough to preserve the conflict but not padded with invented context
- avoids sensitive personal identifiers beyond what is necessary for the
  scenario

Rejected flips should keep the source row in the audit log with a rejection
reason.

## Curation And Sampling

Curation must be completed before any model-under-test runs.

Recommended locked artifacts:

- candidate pool manifest
- rejected candidate manifest with reasons
- accepted pool manifest
- public evaluation sample manifest, for example N=20
- optional larger sample manifest, for example N=100

The official sample should be randomly selected from the accepted pool using a
fixed seed. The seed, sample size, and sampling script hash must be recorded.

Recommended initial sizes:

- accepted public pool: 150 items
- first official run sample: 20 items
- expansion sample: 50 or 100 items, if cost and power justify it

## Paper Reporting

Report the protocol as prospective and separate from official ELEPHANT reuse.
State that:

- the current official comparison uses a curated subset of ELEPHANT-released
  paired flips
- the prospective dataset was defined before model evaluation
- exclusions were based on source quality, privacy, and flip validity, not model
  outcomes
- model outputs were not inspected before finalizing the sample
- all hashes, seeds, and inclusion/exclusion counts are published or preserved
  in an audit bundle

## Current Decision

No prospective data collection has been performed under this protocol. The next
implementation step is to build offline manifest tooling for candidate queues,
curation decisions, and fixed-seed sample selection. Live source access should
remain disabled until compliance and access requirements are satisfied.
