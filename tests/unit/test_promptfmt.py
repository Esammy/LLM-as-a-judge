"""Prompt sectioning and the tolerant JSON extractor."""

from __future__ import annotations

import pytest

from judgekit.core.errors import JudgeParseError
from judgekit.core.promptfmt import TAG_OUTPUT, extract_json, read_section, section


class TestSection:
    def test_roundtrips(self) -> None:
        assert read_section(section(TAG_OUTPUT, "the answer"), TAG_OUTPUT) == "the answer"

    def test_strips_surrounding_whitespace(self) -> None:
        assert read_section(section(TAG_OUTPUT, "\n  padded  \n"), TAG_OUTPUT) == "padded"

    def test_absent_tag_reads_as_none(self) -> None:
        assert read_section("nothing here", TAG_OUTPUT) is None

    def test_handles_multiline_bodies(self) -> None:
        body = "line one\nline two\nline three"
        assert read_section(section(TAG_OUTPUT, body), TAG_OUTPUT) == body

    def test_reads_the_right_tag_among_several(self) -> None:
        text = "\n".join([section("INPUT", "a question"), section(TAG_OUTPUT, "an answer")])
        assert read_section(text, TAG_OUTPUT) == "an answer"


class TestExtractJson:
    """Models wrap JSON in prose and fences however firmly you ask them not to."""

    def test_parses_a_bare_object(self) -> None:
        assert extract_json('{"score": 4}') == {"score": 4}

    def test_tolerates_surrounding_whitespace(self) -> None:
        assert extract_json('\n  {"score": 4}\n ') == {"score": 4}

    def test_parses_a_fenced_block(self) -> None:
        assert extract_json('```json\n{"score": 4}\n```') == {"score": 4}

    def test_parses_an_unlabelled_fence(self) -> None:
        assert extract_json('```\n{"score": 4}\n```') == {"score": 4}

    def test_parses_json_buried_in_prose(self) -> None:
        text = 'Here is my assessment:\n{"score": 4, "reasoning": "fine"}\nHope that helps.'
        assert extract_json(text)["score"] == 4

    def test_prefers_the_outermost_object_when_nested(self) -> None:
        parsed = extract_json('Sure: {"score": 4, "detail": {"inner": 1}} done')
        assert parsed["score"] == 4
        assert parsed["detail"] == {"inner": 1}

    def test_rejects_a_bare_array(self) -> None:
        """A judgement is an object; a list is not a recoverable near-miss."""
        with pytest.raises(JudgeParseError, match="no JSON object found"):
            extract_json("[1, 2, 3]")

    def test_rejects_prose_with_no_json(self) -> None:
        with pytest.raises(JudgeParseError, match="no JSON object found"):
            extract_json("I think it was pretty good, honestly.")

    def test_error_shows_a_preview_of_what_it_got(self) -> None:
        with pytest.raises(JudgeParseError) as exc:
            extract_json("total gibberish from the model")
        assert "total gibberish" in str(exc.value)
