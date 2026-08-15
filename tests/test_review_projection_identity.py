"""Zero-review identity regression for Task 020-T5 (D5 acceptance-critical guard).

The projection layer must leave ``owed_units`` and ``score_rows`` byte-identical
to the pre-projection behaviour when only confirming backfill reviews exist.  The
goldens under ``tests/fixtures/review_projection/`` were captured from the
pre-change code over the backfilled-EPIS fixture; any drift here is a regression.
"""
from __future__ import annotations

import json
from pathlib import Path

from _review_projection_fixture import build_backfilled_epis_run
from suite_tools import owed_units as _owed
from suite_tools import score_rows as _sr

_GOLDEN_DIR = Path(__file__).parent / "fixtures" / "review_projection"


def _normalize(obj, run_dir: str) -> str:
    return json.dumps(obj, indent=2, sort_keys=True).replace(run_dir, "<RUN_DIR>")


def test_owed_units_byte_identical_to_pre_projection(tmp_path):
    run = build_backfilled_epis_run(tmp_path / "run")
    got = _normalize(_owed.owed_units(run, module="epis"), str(run)) + "\n"
    golden = (_GOLDEN_DIR / "owed_units.golden.json").read_text()
    assert got == golden


def test_score_rows_byte_identical_to_pre_projection(tmp_path):
    run = build_backfilled_epis_run(tmp_path / "run")
    got = _normalize(_sr.score_rows(run, module="epis"), str(run)) + "\n"
    golden = (_GOLDEN_DIR / "score_rows.golden.json").read_text()
    assert got == golden
