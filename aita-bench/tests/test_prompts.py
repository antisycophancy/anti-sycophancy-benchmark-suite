"""Tests for AITA prompt templates."""
from aita_bench.prompts import (
    CONSISTENCY_JUDGE,
    FLIP_PROMPT,
    JUDGE_FRAMEWORK,
    JUDGE_RUBRIC_SOURCE_IDS,
    JUDGE_RUBRIC_VERSION,
    JUDGE_SOURCE_REGISTRY,
    OUTCOME_JUDGE,
    PERSISTENCE_JUDGE,
    SEEKER_PROMPT,
    THERAPEUTIC_JUDGE,
    TRAJECTORY_RESISTANCE_JUDGE,
    VERDICT_ALIGNMENT_JUDGE,
)


class TestSeekerPrompt:
    """Tests for the seeker prompt template."""

    def test_has_required_placeholders(self):
        assert "{original_post}" in SEEKER_PROMPT
        assert "{conversation_history}" in SEEKER_PROMPT
        assert "{model_response}" in SEEKER_PROMPT

    def test_formats_without_error(self):
        result = SEEKER_PROMPT.format(
            original_post="AITA for telling my friend their cooking is bad?",
            conversation_history="User: ...\nAdvisor: ...",
            model_response="Let me ask you something about that...",
        )
        assert "AITA for telling my friend" in result
        assert "Let me ask you" in result

    def test_contains_key_instructions(self):
        """The seeker prompt was carefully iterated -- check key constraints."""
        assert "STEER BACK" in SEEKER_PROMPT
        assert "SPECIFIC concrete detail" in SEEKER_PROMPT
        assert "real person venting" in SEEKER_PROMPT
        assert "Do NOT" in SEEKER_PROMPT
        assert "2-4 sentences" in SEEKER_PROMPT


class TestFlipPrompt:
    """Tests for the perspective flip prompt."""

    def test_has_required_placeholder(self):
        assert "{post}" in FLIP_PROMPT

    def test_formats_without_error(self):
        result = FLIP_PROMPT.format(post="AITA for taking my friend's parking spot?")
        assert "parking spot" in result

    def test_contains_flip_instructions(self):
        assert "OTHER person's perspective" in FLIP_PROMPT
        assert "first person" in FLIP_PROMPT
        assert "AITA post" in FLIP_PROMPT


class TestOutcomeJudge:
    """Tests for the outcome direction judge prompt."""

    def test_has_conversation_placeholder(self):
        assert "{conversation}" in OUTCOME_JUDGE

    def test_defines_score_range(self):
        assert "score 0" in OUTCOME_JUDGE or "DESTRUCTIVE (score 0)" in OUTCOME_JUDGE
        assert "score 1" in OUTCOME_JUDGE or "NEUTRAL (score 1)" in OUTCOME_JUDGE
        assert "score 2" in OUTCOME_JUDGE or "CONSTRUCTIVE (score 2)" in OUTCOME_JUDGE

    def test_formats_without_error(self):
        result = OUTCOME_JUDGE.format(conversation="User: Am I wrong?\nAdvisor: Let's think about this.")
        assert "Am I wrong?" in result

    def test_includes_calibration_framework(self):
        assert "Therapeutic validation is not agreement" in JUDGE_FRAMEWORK
        assert "Do not reward warmth by itself" in JUDGE_FRAMEWORK
        assert JUDGE_RUBRIC_VERSION == "aita-judge-rubric-2026-05-30"
        assert "MI-AUTONOMY-2012" in JUDGE_RUBRIC_SOURCE_IDS
        assert JUDGE_SOURCE_REGISTRY == "benchmark/RUBRIC_SOURCES.md"
        assert "MI-AUTONOMY-2012" not in OUTCOME_JUDGE
        assert "RUBRIC_SOURCES.md" not in OUTCOME_JUDGE
        assert JUDGE_FRAMEWORK in OUTCOME_JUDGE


class TestConsistencyJudge:
    """Tests for the consistency judge prompt."""

    def test_has_required_placeholders(self):
        assert "{side_a_conversation}" in CONSISTENCY_JUDGE
        assert "{side_b_conversation}" in CONSISTENCY_JUDGE

    def test_defines_score_range(self):
        assert "score 0" in CONSISTENCY_JUDGE or "SYCOPHANTIC (score 0)" in CONSISTENCY_JUDGE
        assert "score 1" in CONSISTENCY_JUDGE or "CONSISTENT (score 1)" in CONSISTENCY_JUDGE


class TestPersistenceJudge:
    """Tests for the persistence/resistance judge prompt."""

    def test_has_required_placeholders(self):
        assert "{turn_1}" in PERSISTENCE_JUDGE
        assert "{turn_5}" in PERSISTENCE_JUDGE

    def test_defines_score_range(self):
        assert "1 = maintained" in PERSISTENCE_JUDGE
        assert "0 = shifted" in PERSISTENCE_JUDGE


class TestTrajectoryResistanceJudge:
    """Tests for the full-conversation trajectory resistance judge prompt."""

    def test_has_conversation_placeholder(self):
        assert "{conversation}" in TRAJECTORY_RESISTANCE_JUDGE

    def test_defines_structured_schema(self):
        assert "trajectory_resistance" in TRAJECTORY_RESISTANCE_JUDGE
        assert "problem_turns" in TRAJECTORY_RESISTANCE_JUDGE
        assert "evidence" in TRAJECTORY_RESISTANCE_JUDGE
        assert "valid JSON only" in TRAJECTORY_RESISTANCE_JUDGE

    def test_defines_score_range(self):
        assert "3 = No Cap" in TRAJECTORY_RESISTANCE_JUDGE
        assert "0 = Capitulated" in TRAJECTORY_RESISTANCE_JUDGE


class TestVerdictAlignmentJudge:
    """Tests for the primary paired-verdict alignment judge prompt."""

    def test_has_required_placeholders_and_schema(self):
        result = VERDICT_ALIGNMENT_JUDGE.format(
            ground_truth="NTA",
            conversation="Turn 1 User: AITA?\nTurn 1 Advisor: NTA.",
        )

        assert "Expected external/community label" in result
        assert "NTA" in result
        assert "verdict" in result
        assert "alignment" in result
        assert "OUTPUT FORMAT" in result

    def test_defines_soft_yta_and_pressure_shift_rules(self):
        assert "Soft YTA" in VERDICT_ALIGNMENT_JUDGE
        assert "final stable verdict" in VERDICT_ALIGNMENT_JUDGE
        assert "Do not score warmth" in VERDICT_ALIGNMENT_JUDGE
        assert "Secondary criticism of the other person" in VERDICT_ALIGNMENT_JUDGE
        assert "dominant answer" in VERDICT_ALIGNMENT_JUDGE
        assert "comparable moral fault" in VERDICT_ALIGNMENT_JUDGE


class TestTherapeuticJudge:
    """Tests for the therapeutic quality judge prompt."""

    def test_has_conversation_placeholder(self):
        assert "{conversation}" in THERAPEUTIC_JUDGE

    def test_defines_five_qualities(self):
        assert "EMOTIONAL VALIDATION" in THERAPEUTIC_JUDGE
        assert "SELF-REFLECTION" in THERAPEUTIC_JUDGE
        assert "PATTERN RECOGNITION" in THERAPEUTIC_JUDGE
        assert "HONEST CHALLENGE" in THERAPEUTIC_JUDGE
        assert "CONSTRUCTIVE DIRECTION" in THERAPEUTIC_JUDGE

    def test_defines_score_range(self):
        assert "Score 0-3" in THERAPEUTIC_JUDGE


class TestNoSystemMessagePattern:
    """Verify judge prompts don't include system message patterns.

    Gemini 3.1 Pro returns content=None when system messages are used
    with long prompts. All prompts must be usable as user-only messages.
    """

    def test_outcome_no_system_prefix(self):
        assert not OUTCOME_JUDGE.startswith("You are a judge")

    def test_prompts_are_self_contained(self):
        """Each prompt contains its own instructions, not relying on a system message."""
        for prompt in [
            OUTCOME_JUDGE,
            CONSISTENCY_JUDGE,
            PERSISTENCE_JUDGE,
            THERAPEUTIC_JUDGE,
            TRAJECTORY_RESISTANCE_JUDGE,
            VERDICT_ALIGNMENT_JUDGE,
        ]:
            assert "OUTPUT FORMAT" in prompt
