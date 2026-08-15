# AITA Datasets

## Flagship: the sealed construct-reversed set

The default release condition is a separately signed sealed data pack with 20
human-reviewed pairs. The public software tree intentionally contains no
Reddit-derived prompt text, source URLs, label sidecar, or selection. The pack
is versioned independently so a takedown or correction can remove or supersede
that distribution without rewriting the benchmark software.

This is public anti-indexing friction, not confidentiality, DRM, or access
control. Part A is embedded in the envelope and Part B is published in the
signed suite release as a separate asset outside Git history. The runner decrypts only in memory and verifies the pack,
embedded files, selection, and per-pair identities before provider spend. See
`curated/aita_reversed_n20_v1/PACK.md` and
`../../manifests/aita-sealed-pack-v1.json`; `SCORING.md` documents the public
two-axis scoring contract.

The repository's MIT license applies to project code, not automatically to
third-party source posts. Read [`../../docs/DATA_RIGHTS_AND_PRIVACY.md`](../../docs/DATA_RIGHTS_AND_PRIVACY.md)
before redistributing the dataset or result bundles.

`../../manifests/aita-data-clearance.json` records that Reddit-derived
plaintext is excluded from the software release. The external sealed pack has
its own immutable registry identity and release review. Send correction or
removal requests to `research@antisycophancy.ai` with only the Reddit post ID or
pack item ID.

```bash
aita-bench run --dataset-mode nta-paired \
  --sealed-pack /absolute/path/aita-reversed-n20-v1.envelope.json \
  --items 20 --model <model> --output <dir>
```

The command prompts invisibly for Part B. Noninteractive automation must opt in
with `--sealed-key-part-b-from-env`; Part B must never be placed directly on the
command line. The authenticated label policy remains explicit: ESH labels are
excluded from verdict-alignment while consistency (both-NTA = failure) applies
to every item.

## Advanced / maintainer: ELEPHANT-derived modes

The benchmark *design* adapts the [ELEPHANT benchmark](https://arxiv.org/abs/2505.13995)
(Stanford/CMU/Oxford, ICLR 2026) AITA setting. Two ELEPHANT-data modes remain in
the runner for advanced/maintainer use, but the CC0 source CSVs and the fetch
tool are **not shipped in the public repo** (they live in the maintainer's
private tree):

- **`yta-synthflip`** — reads `AITA-YTA.csv` (clear-cut YTA posts) and generates
  a synthetic side-B perspective at runtime. The original auto-generated path;
  superseded by the curated set because hand-reviewed reversals are higher
  quality.
- **`nta-paired` on official data** — `AITA-NTA-OG.csv` + `AITA-NTA-FLIP.csv`
  (official human flips), merged by `id`. Requires a `--paired-labels` answer key
  like any nta-paired set.

To use these, fetch the CC0 ELEPHANT data from
[OSF](https://osf.io/r3dmj/) (maintainer fetch script at
`internal/maintainer_tools/aita_elephant_fetch/`). `AITA-YTA_sample.csv` (three
synthetic rows) ships for smoke tests via `--allow-sample-fallback`.

## Provenance discipline (all modes)

Source CSVs are never edited in place. AITA run contracts record an
`aita-dataset-manifest-v1` manifest with per-file source hashes, the label
policy digest, selected pair IDs, malformed rows, and per-pair
`source_pair_hash` + per-side prompt hashes — so runs are bound to the exact
items and labels, and overlapping questions stay comparable across refreshes.
If a cleaned derivative is ever needed, publish it as a distinct, versioned
dataset (e.g. `aita_reversed_n20_v2`) rather than mutating an existing one.
