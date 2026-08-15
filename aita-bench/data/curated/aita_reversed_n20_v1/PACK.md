# AITA Reversed N20 v1

This directory intentionally contains no Reddit-derived prompt text, source
URLs, label sidecar, or locked selection. The runnable N=20 condition is
distributed as a separately signed sealed data-pack release. The encryption is
publicly reversible anti-indexing friction, not confidentiality, DRM, or an
access-control claim.

The suite decrypts the pack only in memory. It verifies the authenticated
envelope, ciphertext digest, exact embedded-file hashes, locked selection, and
per-pair prompt identities before a prepared run may spend against a provider.
Plaintext source files are not written into the software checkout.

The canonical public identity, migration hashes, expected filenames, and key
split locations are recorded in
[`manifests/aita-sealed-pack-v1.json`](../../../../manifests/aita-sealed-pack-v1.json).
Once the final repository and release URLs are approved, that registry entry
will point to the independently signed data-pack release. Part A is embedded in
the pack envelope. Part B is published as a separate asset attached to the
signed suite release and is not tracked in Git.

Prepare a run after downloading the envelope and adjacent ciphertext:

```bash
./venv/bin/python -m suite_tools.prepare_run --module aita \
  --run-id aita-n20-v1 --models <selector> --judge-set frontier \
  --sealed-pack /absolute/path/aita-reversed-n20-v1.envelope.json \
  --output results/prepared/aita-n20-v1
```

The command prompts invisibly for Part B. For deliberate noninteractive use,
set `ANTISYCOPHANCY_AITA_PACK_KEY_PART_B` and add
`--sealed-key-part-b-from-env`; the runner consumes that variable and does not
serialize it into the command or run contract. Never put Part B directly on a
command line.

See [SCORING.md](SCORING.md) for the public scoring contract and
[`docs/DATA_RIGHTS_AND_PRIVACY.md`](../../../../docs/DATA_RIGHTS_AND_PRIVACY.md)
for the rights, privacy, correction, and removal policy. Contact
`research@antisycophancy.ai` with only a pack item ID or source post ID.
