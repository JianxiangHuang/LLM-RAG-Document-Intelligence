from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.language import Language

from services.count_token_service import count_tokens

MAX_ATOMIC_UNIT_CHARACTERS = 500

CHINESE_SENTENCE_TERMINATORS = {"。", "！", "？"}
ENGLISH_SENTENCE_TERMINATORS = {".", "!", "?"}

CLOSING_CHARACTERS = {
    '"',
    "'",
    "”",
    "’",
    "）",
    "】",
    "》",
    ")",
    "]",
    "}",
}


@lru_cache(maxsize=1)
def _get_english_nlp() -> Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError as error:
        raise RuntimeError(
            "spaCy English model 'en_core_web_sm' is not installed. "
            "Install the project dependencies before running the splitter."
        ) from error


def _build_unit(
        original_text: str,
        start_char: int,
        end_char: int,
        unit_type: str,
) -> dict:
    """
    Create a unit that can always be mapped back to original_text.
    """
    text = original_text[start_char:end_char]

    return {
        "type": unit_type,
        "text": text,
        "start_char": start_char,
        "end_char": end_char,
        "count": end_char - start_char,
        "token_count": count_tokens(text),
        "metadata": {
            "source_position":{
                "type":"text_offset",
                "start_char":start_char,
                "end_char":end_char,
            }
        },
    }


def _trim_span(
        original_text: str,
        start_char: int,
        end_char: int,
) -> tuple[int, int]:
    segment = original_text[start_char:end_char]

    leading_whitespace = len(segment) - len(segment.lstrip())
    trailing_whitespace = len(segment) - len(segment.rstrip())

    return (
        start_char + leading_whitespace,
        end_char - trailing_whitespace,
    )


def split_atomic_units(original_text: str) -> list[dict]:
    """
    Split text into atomic units while preserving exact original spans.

    Rules:
    - Newlines force a boundary.
    - Chinese 。！？ force a sentence boundary.
    - English .!? boundaries are determined by spaCy.
    - Closing quotes/brackets after a sentence terminator are included in the preceding unit.
    - Text without a recognized sentence boundary longer than 500 characters becomes
      length_fallback units.
    """
    if not isinstance(original_text, str):
        raise TypeError("Original text must be a string.")

    if original_text == "":
        return []

    units: list[dict] = []
    text_length = len(original_text)

    line_start = 0
    index = 0

    while index < text_length:
        if original_text[index] not in {"\n", "\r"}:
            index += 1
            continue

        _append_line_units(
            original_text=original_text,
            line_start=line_start,
            line_end=index,
            units=units,
        )

        # Treat Windows CRLF as one newline boundary.
        if (
                original_text[index] == "\r"
                and index + 1 < text_length
                and original_text[index + 1] == "\n"
        ):
            index += 1

        line_start = index + 1
        index += 1

    # Process the final line, including a document with no newline.
    _append_line_units(
        original_text=original_text,
        line_start=line_start,
        line_end=text_length,
        units=units,
    )

    return units


def _append_line_units(
        original_text: str,
        line_start: int,
        line_end: int,
        units: list[dict],
) -> None:
    """
    Process one newline-delimited line.

    Chinese terminators are forced boundaries.
    English sentence boundaries come from spaCy.
    """
    line_start, line_end = _trim_span(
        original_text,
        line_start,
        line_end,
    )

    if line_start >= line_end:
        return

    line_text = original_text[line_start:line_end]
    english_nlp = _get_english_nlp()
    document = english_nlp(line_text)

    # spaCy offsets are relative to line_text, so convert them back
    # to absolute offsets in original_text.
    spacy_sentence_ends = {
        line_start + sentence.end_char
        for sentence in document.sents
    }

    chinese_sentence_ends = {
        index + 1
        for index in range(line_start, line_end)
        if original_text[index] in CHINESE_SENTENCE_TERMINATORS
    }

    candidate_ends = sorted(
        spacy_sentence_ends | chinese_sentence_ends
    )

    current_start = line_start

    for raw_end in candidate_ends:
        if raw_end <= current_start:
            continue

        raw_end = min(raw_end, line_end)
        end_char = _consume_closing_characters(
            original_text,
            raw_end,
            line_end,
        )

        start_char, end_char = _trim_span(
            original_text,
            current_start,
            end_char,
        )

        if start_char >= end_char:
            current_start = max(current_start, end_char)
            continue

        is_chinese_sentence = raw_end in chinese_sentence_ends
        is_english_sentence = (
                raw_end in spacy_sentence_ends
                and _ends_with_english_terminator(
            original_text,
            start_char,
            raw_end,
        )
        )

        if is_chinese_sentence or is_english_sentence:
            units.append(
                _build_unit(
                    original_text=original_text,
                    start_char=start_char,
                    end_char=end_char,
                    unit_type="sentence",
                )
            )
        else:
            _append_fallback_units(
                original_text=original_text,
                start_char=start_char,
                end_char=end_char,
                units=units,
            )

        current_start = max(current_start, end_char)

    # The remaining text has no recognized sentence boundary.
    if current_start < line_end:
        _append_fallback_units(
            original_text=original_text,
            start_char=current_start,
            end_char=line_end,
            units=units,
        )


def _consume_closing_characters(
        original_text: str,
        end_char: int,
        line_end: int,
) -> int:
    """
    Include trailing quotes/brackets in the preceding sentence.

    Example:
    He said "Done." Next sentence.
    -> He said "Done."
    """
    while (
            end_char < line_end
            and original_text[end_char] in CLOSING_CHARACTERS
    ):
        end_char += 1

    return end_char


def _ends_with_english_terminator(
        original_text: str,
        start_char: int,
        end_char: int,
) -> bool:
    """
    Check whether the spaCy sentence span actually ends in . ! or ?.

    spaCy returns a sentence span even for text without punctuation,
    such as a heading. Such text should remain paragraph_fallback.
    """
    index = end_char - 1

    while (
            index >= start_char
            and original_text[index] in CLOSING_CHARACTERS
    ):
        index -= 1

    return (
            index >= start_char
            and original_text[index] in ENGLISH_SENTENCE_TERMINATORS
    )


def _append_fallback_units(
        original_text: str,
        start_char: int,
        end_char: int,
        units: list[dict],
) -> None:
    """
    Add paragraph_fallback or length_fallback units.

    A fallback span longer than 500 characters is split at the nearest
    whitespace at or before the limit. If no whitespace exists, it is
    hard-cut at 500 characters.
    """
    start_char, end_char = _trim_span(
        original_text,
        start_char,
        end_char,
    )

    if start_char >= end_char:
        return

    if end_char - start_char <= MAX_ATOMIC_UNIT_CHARACTERS:
        units.append(
            _build_unit(
                original_text=original_text,
                start_char=start_char,
                end_char=end_char,
                unit_type="paragraph",
            )
        )
        return

    current_start = start_char

    while end_char - current_start > MAX_ATOMIC_UNIT_CHARACTERS:
        hard_end = current_start + MAX_ATOMIC_UNIT_CHARACTERS
        split_at = _find_fallback_split_position(
            original_text,
            current_start,
            hard_end,
        )
        _, split_at = _trim_span(
            original_text,
            current_start,
            split_at,
        )

        units.append(
            _build_unit(
                original_text=original_text,
                start_char=current_start,
                end_char=split_at,
                unit_type="length_fallback",
            )
        )

        segment = original_text[split_at:end_char]
        leading_whitespace = len(segment) - len(segment.lstrip())
        current_start = split_at + leading_whitespace

    if current_start < end_char:
        units.append(
            _build_unit(
                original_text=original_text,
                start_char=current_start,
                end_char=end_char,
                unit_type="length_fallback",
            )
        )


def _find_fallback_split_position(
        original_text: str,
        start_char: int,
        hard_end: int,
) -> int:
    """
    Find the rightmost whitespace at or before hard_end.

    The whitespace itself is excluded from the preceding unit.
    """
    for index in range(hard_end, start_char, -1):
        if original_text[index].isspace():
            return index

    return hard_end
