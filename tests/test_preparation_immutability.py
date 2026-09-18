import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from suite_tools.prepare_run import prepare_aita_run, prepare_epis_run, prepare_sus_run


def _prepare(root, module):
    common = dict(run_id="frozen", output_root=root, suite_config_path=Path("suite_models.yaml"),
                  model_selector="group:calibration_smoke", judge_set="calibration")
    if module == "sus":
        return prepare_sus_run(**common, scenarios_selector="bridge_heights", runs=1)
    if module == "aita":
        return prepare_aita_run(**common, items="1", dataset_mode="yta-synthflip", allow_sample_fallback=True)
    return prepare_epis_run(**common, items=1, types="delusion")


def _snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("module", ["sus", "aita", "epis"])
def test_reprepare_refuses_before_changing_any_existing_run_file(tmp_path, module):
    contract = _prepare(tmp_path, module)
    assert json.loads(contract.read_text())["artifact_protocol_version"] == "benchmark-artifact-protocol-v1"
    (contract.parent / "RUN_STATUS.json").write_text(json.dumps({"status": "completed", "validity": "score_ready"}))
    (contract.parent / "RUN_EVENTS.jsonl").write_text('{"event":"stage_completed"}\n')
    before = _snapshot(tmp_path)
    with pytest.raises(FileExistsError, match="new.*output|already exists"):
        _prepare(tmp_path, module)
    assert _snapshot(tmp_path) == before


def test_preparation_claim_is_exclusive_between_contenders(tmp_path):
    def attempt():
        try:
            return _prepare(tmp_path, "sus")
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))
    assert sum(path is not None for path in outcomes) == 1
    assert json.loads((tmp_path / "sus" / "RUN_CONTRACT.json").read_text())["lifecycle_state"] == "prepared"


def test_preparation_still_allows_a_different_module_in_group(tmp_path):
    sus = _prepare(tmp_path, "sus")
    before = sus.read_bytes()
    epis = _prepare(tmp_path, "epis")
    assert sus.read_bytes() == before
    assert epis.is_file()


def test_orphaned_rendered_config_is_not_overwritten(tmp_path):
    config = tmp_path / "_configs" / "calibration" / "sus-models.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("historical: keep\n")
    before = _snapshot(tmp_path)
    with pytest.raises(FileExistsError):
        _prepare(tmp_path, "sus")
    assert _snapshot(tmp_path) == before
