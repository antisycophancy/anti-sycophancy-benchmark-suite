# Public Results Viewers

The public results viewer is a reproducible presentation layer over saved
benchmark artifacts. It does not run models, score transcripts, or change
stored results.

Use it for paper/blog/website evidence pages where readers need a compact score
map and a transcript reader with keyboard navigation.

## Generate A Viewer

From the benchmark repo:

```bash
cd /path/to/benchmark
```

Generate the current Opus native-effort N=10 targeted SUS page:

```bash
./venv/bin/python -m suite_tools.public_results_page \
  results/dashboard-watch/anthropic-native-effort-sus-n10-targeted-combined-20260528-224447/sus \
  --suite sus \
  --title "SUS Opus Native Effort Results" \
  --output results/drafts/sus-opus-native-effort-results.html
```

Generate the Opus native-effort N=5 full matrix page, including Opus 4.6:

```bash
./venv/bin/python -m suite_tools.public_results_page \
  results/dashboard-watch/anthropic-native-effort-sus-n5-combined-20260528-204358/sus \
  --suite sus \
  --title "SUS Opus Native Effort N5 Results" \
  --output results/drafts/sus-opus-native-effort-n5-results.html
```

Generate the default multi-suite draft:

```bash
./venv/bin/python -m suite_tools.public_results_page \
  --title "Benchmark Results" \
  --output results/drafts/public-benchmark-results.html
```

`results/` is intentionally ignored by git. Commit the generator, runbooks, and
tests; treat generated HTML and raw result JSON as evidence artifacts that can
be copied into a website build or archival bundle when the release set is
final. The reusable viewer template lives in `suite_tools/public_results_page.py`
and is covered by `tests/test_public_results_page.py`; generated pages under
`results/drafts/` are examples or release artifacts, not the source template.

## Display Rules

- Compact model codes follow `MODEL_NOMENCLATURE.md`.
- SUS pages group records by model condition, then place repeated runs inside
  that condition.
- AITA red/green status is verdict-first: `verdict No Cap` means the model gave
  the expected side verdict for that paired moral flip, while `verdict Cap`
  means the model crossed to the wrong side. Outcome, resistance, trajectory,
  and therapeutic dimensions remain visible as diagnostic quality flags, but
  they do not by themselves turn a verdict-correct AITA record red.
- AITA notes should surface judge verdict rationale/evidence where available so
  readers and model providers can understand why a response was classified as
  YTA, NTA, MIXED, or unclear.
- Model condition cards show the model name first and effort/config on the next
  line, for example `Claude Opus 4.8` plus `native effort high`.
- The transcript selector uses the compact model code and the effort/config
  label so same-model comparisons are visible without opening each item.
- For Opus 4.8, `high` is the default baseline effort for publication runs.
  `xhigh`/extra and `max` should be labeled as non-default exploratory or
  stress-test conditions unless the run plan explicitly promotes them.
- Arrow keys move through viewer items when the transcript viewer is focused.

## Website Integration

For the anti-sycophancy website, prefer one of these paths:

1. Copy a reviewed generated HTML page into the website as a static evidence
   page.
2. Reuse the generated `public-results-data` JSON payload and the viewer
   components/styles inside the website app.
3. Generate the page during a release script from archived result artifacts.

The website should not recompute scores. It should consume contract-backed,
hygiene-checked artifacts and render them read-only.

When adapting the template into the public site, keep the benchmark semantics
the same even if the visual brand changes: infrastructure artifacts are gold and
excluded, SUS red means safety-boundary capitulation, AITA red means verdict
capitulation, and Epistemic red means evidence/stance drift according to its
rubric.
