"""Preprocessing utilities for end-of-turn detection."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass


QUESTION_WORDS = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
    "am",
}

BACKCHANNELS = {
    "yeah",
    "yes",
    "yep",
    "okay",
    "ok",
    "right",
    "sure",
    "mhm",
    "uh-huh",
    "thanks",
    "thank you",
}

INCOMPLETE_CUES = {
    "to",
    "and",
    "or",
    "but",
    "because",
    "that",
    "if",
    "when",
    "while",
    "for",
    "with",
    "about",
    "maybe",
    "like",
    "mean",
}

HESITATION_CUES = {"um", "uh", "erm", "well", "so", "i mean", "you know", "like"}


@dataclass(frozen=True)
class TextSignals:
    """Lightweight linguistic signals extracted from a partial transcript."""

    num_words: int
    last_token: str
    ends_with_punctuation: int
    has_question_word: int
    syntactic_completeness_score: float
    is_backchannel: int
    is_interruption: int


def normalize_text(text: str) -> str:
    """Return lower-cased text with extra whitespace removed."""

    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokenize(text: str) -> list[str]:
    """Tokenize text with a simple word-oriented regex."""

    return re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?", normalize_text(text))


def get_last_token(text: str) -> str:
    tokens = tokenize(text)
    return tokens[-1] if tokens else ""


def has_sentence_final_punctuation(text: str) -> int:
    return int(str(text).strip().endswith((".", "?", "!")))


def has_question_word(text: str) -> int:
    tokens = tokenize(text)
    return int(bool(tokens) and (tokens[0] in QUESTION_WORDS or "?" in str(text)))


def is_backchannel(text: str) -> int:
    cleaned = normalize_text(text).strip(string.punctuation + " ")
    return int(cleaned in BACKCHANNELS)


def is_interruption(text: str) -> int:
    stripped = str(text).strip()
    return int(stripped.endswith(("-", "--", "...")) or "[interrupt" in stripped.lower())


def syntactic_completeness_score(text: str) -> float:
    """Estimate whether a partial transcript looks syntactically complete.

    This is intentionally lightweight. It is not a grammar parser; it gives the
    classifier a transparent proxy feature for portfolio-scale experimentation.
    """

    cleaned = normalize_text(text)
    tokens = tokenize(cleaned)
    if not tokens:
        return 0.0

    score = 0.45
    last = tokens[-1]

    if has_sentence_final_punctuation(text):
        score += 0.35
    if has_question_word(text):
        score += 0.12
    if is_backchannel(text):
        score += 0.25
    if len(tokens) >= 5:
        score += 0.08
    if last in INCOMPLETE_CUES:
        score -= 0.35
    if any(cue in cleaned for cue in HESITATION_CUES):
        score -= 0.08
    if is_interruption(text):
        score -= 0.30

    return round(min(max(score, 0.0), 1.0), 2)


def extract_text_signals(text: str) -> TextSignals:
    tokens = tokenize(text)
    return TextSignals(
        num_words=len(tokens),
        last_token=get_last_token(text),
        ends_with_punctuation=has_sentence_final_punctuation(text),
        has_question_word=has_question_word(text),
        syntactic_completeness_score=syntactic_completeness_score(text),
        is_backchannel=is_backchannel(text),
        is_interruption=is_interruption(text),
    )
