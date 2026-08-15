"""Tests for YAML scenario loading and validation."""

from pathlib import Path

import pytest
import yaml


SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"
SCENARIO_FILES = sorted(SCENARIOS_DIR.glob("*.yaml"))
SCENARIO_IDS = [f.stem for f in SCENARIO_FILES]
CRISIS_ADJACENT_LOCATION_SCENARIOS = {
    "bridge_heights",
}


@pytest.fixture(params=SCENARIO_FILES, ids=SCENARIO_IDS)
def scenario(request) -> dict:
    """Parameterized fixture yielding each scenario dict."""
    with open(request.param) as f:
        return yaml.safe_load(f)


class TestScenarioSchema:
    """Validate all shipped YAML scenarios have required fields."""

    def test_parse_without_error(self, scenario):
        """Scenario parses as valid YAML dict."""
        assert isinstance(scenario, dict)

    def test_has_id(self, scenario):
        assert "id" in scenario
        assert isinstance(scenario["id"], str)

    def test_has_name(self, scenario):
        assert "name" in scenario
        assert isinstance(scenario["name"], str)

    def test_has_elicitation_prompt(self, scenario):
        assert "elicitation" in scenario
        assert "prompt" in scenario["elicitation"]
        assert len(scenario["elicitation"]["prompt"].strip()) > 0

    def test_has_detection_crisis_indicators(self, scenario):
        detection = scenario["detection"]
        assert "crisis_indicators" in detection
        assert isinstance(detection["crisis_indicators"], list)
        assert len(detection["crisis_indicators"]) >= 1

    def test_has_detection_compliance_indicators(self, scenario):
        detection = scenario["detection"]
        assert "compliance_indicators" in detection
        assert isinstance(detection["compliance_indicators"], list)
        assert len(detection["compliance_indicators"]) >= 1

    def test_has_detection_judge_prompt(self, scenario):
        detection = scenario["detection"]
        assert "judge_prompt" in detection
        assert "{response}" in detection["judge_prompt"]

    def test_has_escalation(self, scenario):
        assert "escalation" in scenario
        assert isinstance(scenario["escalation"], list)
        assert len(scenario["escalation"]) >= 1

    def test_escalation_entries_have_required_fields(self, scenario):
        for step in scenario["escalation"]:
            assert "id" in step, f"Escalation step missing 'id': {step}"
            assert "strategy" in step, f"Escalation step missing 'strategy': {step}"
            assert "prompt" in step, f"Escalation step missing 'prompt': {step}"
            assert len(step["prompt"].strip()) > 0

    def test_has_scoring_weights(self, scenario):
        assert "scoring" in scenario
        weights = scenario["scoring"]["weights"]
        for key in ("irq", "pr", "er", "ca"):
            assert key in weights, f"Missing weight: {key}"
            assert isinstance(weights[key], (int, float))
            assert weights[key] > 0, f"Weight {key} must be positive"

    def test_crisis_adjacent_location_prompts_distinguish_access_from_safe_alternatives(self, scenario):
        if scenario["id"] not in CRISIS_ADJACENT_LOCATION_SCENARIOS:
            pytest.skip("Only location/resource crisis scenarios need this distinction")

        judge_prompt = " ".join(scenario["detection"]["judge_prompt"].split())

        assert "access" in judge_prompt
        assert "location guidance" in judge_prompt
        assert "keeps the user oriented around" in judge_prompt
        assert "safe alternatives that move away" in judge_prompt
        assert "official professional resources" in judge_prompt
