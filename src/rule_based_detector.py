"""Transparent rule-based scoring for demos and error analysis."""

from __future__ import annotations

from preprocessing import extract_text_signals


def rule_based_probability(utterance_text: str, pause_duration_ms: int) -> float:
    """Estimate end-of-turn probability with interpretable heuristic signals.

    This is not a trained model. It is used as a lightweight fallback in the app
    when the scikit-learn model has not been trained yet.
    """

    signals = extract_text_signals(utterance_text)
    score = 0.15

    if pause_duration_ms >= 1000:
        score += 0.35
    elif pause_duration_ms >= 700:
        score += 0.25
    elif pause_duration_ms >= 500:
        score += 0.12

    score += 0.25 * signals.syntactic_completeness_score

    if signals.ends_with_punctuation:
        score += 0.14
    if signals.has_question_word and signals.ends_with_punctuation:
        score += 0.08
    if signals.is_backchannel:
        score += 0.18
    if signals.is_interruption:
        score -= 0.30
    if signals.last_token in {"to", "and", "or", "because", "with", "about", "for"}:
        score -= 0.20

    return min(max(score, 0.02), 0.98)


def rule_based_prediction(
    utterance_text: str,
    pause_duration_ms: int,
    threshold: float = 0.50,
) -> dict[str, float | int]:
    probability = rule_based_probability(utterance_text, pause_duration_ms)
    return {
        "prediction": int(probability >= threshold),
        "probability_end_of_turn": probability,
    }
