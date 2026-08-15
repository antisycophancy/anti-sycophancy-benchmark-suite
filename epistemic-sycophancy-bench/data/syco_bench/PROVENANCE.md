# Vendored Syco-Bench Source Items

These CSVs are vendored verbatim from upstream Syco-Bench so that item
selection is pinned: the benchmark selects items by **CSV row index**
(`data/selection.yaml`), so any upstream reorder or edit would silently
change item identity. Vendoring the files freezes them.

| Field | Value |
|-------|-------|
| Upstream | https://github.com/timfduffy/syco-bench |
| Commit | `5ecc005b2a8cd429c5073763cd4714043123c2cd` |
| Upstream path | `prompts_en/questions/` |
| License | MIT No Attribution (see `LICENSE`, copied from upstream) |
| Vendored | 2026-06-10 |

## Files

| File | sha256 |
|------|--------|
| `delusion.csv` | `fbf614631b4a6202649554205ada62e7cf9cd12d0d1ce019ad683c82cfa7d236` |
| `pickside.csv` | `128633e1c7090c1b6ab36abaee4a6275698d6967a23f350e0112ecd0e8d9423b` |
| `mirror.csv` | `656647d1101984216d8c19bd65990f0d778880e673cc956ef30761420c2a4186` |

These hashes are enforced by `tests/test_runner.py::TestVendoredSycoBenchData`.
Do not edit these files; to adopt newer upstream content, vendor it as a new
versioned selection (new files + new `selection.yaml`) so existing published
results keep their item identity.
