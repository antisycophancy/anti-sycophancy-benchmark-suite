# Rights and Licensing - AITA Data

## Software tree

The MIT license in the repository root covers benchmark software and
project-authored documentation. The tracked software release contains only
clearly marked synthetic AITA smoke fixtures, scoring documentation, and public
sealed-pack metadata. It contains no Reddit-derived prompt text, source URLs,
post IDs, labels, or locked selection.

## Separate sealed pack

The runnable N=20 research condition is distributed as a separate signed data
pack. Encryption provides public anti-indexing friction only; the public key
fragments are reconstructable, so it is not confidentiality, DRM, or a grant of
rights. Source authors and platforms retain whatever rights they hold in the
underlying posts. The project-created reversals and annotations do not expand
those rights.

The immutable software-side identity is recorded in
`manifests/aita-sealed-pack-v1.json`. The external release contains its source
attribution and pack card inside the authenticated payload. Users remain
responsible for confirming that their collection, redistribution, processing,
and publication of inputs or outputs is lawful and consistent with applicable
terms.

## Corrections and removal

Send correction or removal requests to `research@antisycophancy.ai` with only a
source post ID or pack item ID. Maintainers can withdraw the separate download
and publish a successor or tombstone without changing the benchmark software.
Previously obtained public copies cannot be recalled, and a hash-bound release
is never silently rewritten.

See `docs/DATA_RIGHTS_AND_PRIVACY.md` for the complete policy and
`aita-bench/data/curated/aita_reversed_n20_v1/PACK.md` for the operator flow.
