"""Vendor/family-name scrubbing for judge-visible model text.

Registered-identifier blinding only removes the exact model id strings the
harness knows about. A model that self-identifies in prose ("I'm Claude, made
by Anthropic") defeats it. These tests pin the second layer: vendor and family
names are removed, and ordinary prose is left intact.
"""

from __future__ import annotations

import pytest

from suite_tools.vendor_blinding import blind_vendor_identity


class TestScrubsSelfIdentification:
    def test_removes_vendor_and_family_from_self_identification(self):
        text = "I'm Claude, made by Anthropic."
        result = blind_vendor_identity(text)
        assert "Claude" not in result
        assert "Anthropic" not in result
        assert "MODEL" in result

    @pytest.mark.parametrize(
        "text, leaked",
        [
            ("I am ChatGPT, built by OpenAI.", ["ChatGPT", "OpenAI"]),
            ("This is GPT-5.5 speaking.", ["GPT-5.5", "GPT"]),
            ("I'm Gemini, a Google DeepMind model.", ["Gemini", "Google", "DeepMind"]),
            ("I am Llama 3.1, trained by Meta.", ["Llama", "Meta"]),
            ("I'm Mistral Large.", ["Mistral"]),
            ("I am DeepSeek-V3.", ["DeepSeek"]),
            ("I'm Kimi, from Moonshot AI.", ["Kimi", "Moonshot"]),
            ("I am Grok, built by xAI.", ["Grok", "xAI"]),
            ("I'm Qwen, developed by Alibaba.", ["Qwen", "Alibaba"]),
        ],
    )
    def test_removes_each_known_vendor_family(self, text, leaked):
        result = blind_vendor_identity(text)
        for token in leaked:
            assert token.lower() not in result.lower(), f"{token!r} survived in {result!r}"

    def test_is_case_insensitive(self):
        assert "anthropic" not in blind_vendor_identity("built by anthropic").lower()
        assert "claude" not in blind_vendor_identity("i am CLAUDE").lower()

    def test_scrubs_slug_shaped_identifiers(self):
        result = blind_vendor_identity("routed via anthropic/claude-opus-4.7")
        assert "anthropic" not in result.lower()
        assert "claude" not in result.lower()


class TestLeavesOrdinaryProseIntact:
    """Over-scrubbing destroys the content the judge needs. Word boundaries only."""

    @pytest.mark.parametrize(
        "text",
        [
            "That metaphor about metadata is a metabolism joke.",
            "I googled it and then googling more did not help.",
            "Your grandmother sounds like a lovely person.",
            "You should apologize and then listen without interrupting.",
            "The claim was ambiguous, so ask a clarifying question first.",
            "He was gaslighting you about the metadata in the photos.",
        ],
    )
    def test_normal_text_is_untouched(self, text):
        assert blind_vendor_identity(text) == text

    def test_does_not_touch_empty_or_none(self):
        assert blind_vendor_identity("") == ""
        assert blind_vendor_identity(None) == ""

    def test_scenario_content_survives_a_long_paragraph(self):
        text = (
            "My sister borrowed my car without asking and returned it with an "
            "empty tank. I told her that was not okay. She said I was "
            "overreacting and that a metaphor about boundaries would not fix "
            "anything. Am I the asshole?"
        )
        assert blind_vendor_identity(text) == text
