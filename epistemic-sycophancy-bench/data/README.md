# Epistemic Sycophancy Data

Source data comes from Syco-Bench (MIT-0), vendored verbatim in
`syco_bench/` — see `syco_bench/PROVENANCE.md` for the pinned upstream
commit and content hashes. Items are selected by CSV row index
(`selection.yaml`), so the source files are frozen in-repo; no external
clone is required.

- `delusion.csv` — 20 delusional statements (single `statement` column)
- `pickside.csv` — 40 debate pairs (`statement1`, `statement2`)
- `mirror.csv` — 40 opinion topics (`topic`, `stance1`, `stance2`, `userstance1`, `userstance2`)

These are used as source material for our multi-turn adaptive wrappers.
