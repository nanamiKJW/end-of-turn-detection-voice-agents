"""Prediction utilities for the Streamlit app and CLI."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import joblib
import pandas as pd

from preprocessing import extract_text_signals
from rule_based_detector import rule_based_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "end_of_turn_classifier.joblib"
LIGHTWEIGHT_MODEL_PATH = PROJECT_ROOT / "models" / "lightweight_eot_model.json"


def tokenize_for_lightweight_model(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?", text.lower())


def gaussian_log_probability(value: float, mean: float, std: float) -> float:
    variance = std * std
    return -0.5 * math.log(2 * math.pi * variance) - ((value - mean) ** 2 / (2 * variance))


def predict_with_lightweight_model(
    utterance_text: str,
    pause_duration_ms: int,
    model_path: Path = LIGHTWEIGHT_MODEL_PATH,
) -> dict[str, object]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    signals = extract_text_signals(utterance_text)
    numeric_row = {
        "pause_duration_ms": float(pause_duration_ms),
        "speech_duration_ms": float(max(250, signals.num_words * 360)),
        "num_words": float(signals.num_words),
        "ends_with_punctuation": float(signals.ends_with_punctuation),
        "has_question_word": float(signals.has_question_word),
        "syntactic_completeness_score": float(signals.syntactic_completeness_score),
        "is_backchannel": float(signals.is_backchannel),
        "is_interruption": float(signals.is_interruption),
    }

    tokens = tokenize_for_lightweight_model(utterance_text)
    vocabulary_size = int(model["vocabulary_size"])
    scores = {}

    for label in ("0", "1"):
        score = math.log(float(model["class_priors"][label]))
        token_counts = model["token_counts"][label]
        token_total = int(model["token_totals"][label])

        for token in tokens:
            count = int(token_counts.get(token, 0))
            score += math.log((count + 1) / (token_total + vocabulary_size))

        for feature in model["numeric_features"]:
            stats = model["numeric_stats"][label][feature]
            score += gaussian_log_probability(
                numeric_row[feature],
                float(stats["mean"]),
                float(stats["std"]),
            )
        scores[label] = score

    max_score = max(scores.values())
    prob_1 = math.exp(scores["1"] - max_score)
    prob_0 = math.exp(scores["0"] - max_score)
    probability_end = prob_1 / (prob_0 + prob_1)

    return {
        "prediction": int(probability_end >= 0.5),
        "probability_end_of_turn": probability_end,
        "signals": signals,
        "baseline_700ms": int(pause_duration_ms >= 700),
        "model_type": "lightweight_trained_model",
    }


def build_prediction_row(
    utterance_text: str,
    pause_duration_ms: int,
    speech_duration_ms: int | None = None,
) -> pd.DataFrame:
    signals = extract_text_signals(utterance_text)
    estimated_speech_duration = speech_duration_ms or max(250, signals.num_words * 360)
    return pd.DataFrame(
        [
            {
                "utterance_text": utterance_text,
                "pause_duration_ms": pause_duration_ms,
                "speech_duration_ms": estimated_speech_duration,
                "num_words": signals.num_words,
                "ends_with_punctuation": signals.ends_with_punctuation,
                "has_question_word": signals.has_question_word,
                "syntactic_completeness_score": signals.syntactic_completeness_score,
                "is_backchannel": signals.is_backchannel,
                "is_interruption": signals.is_interruption,
            }
        ]
    )


def predict_end_of_turn(
    utterance_text: str,
    pause_duration_ms: int,
    model_path: Path = MODEL_PATH,
    use_fallback: bool = True,
) -> dict[str, object]:
    if not model_path.exists() and LIGHTWEIGHT_MODEL_PATH.exists():
        return predict_with_lightweight_model(utterance_text, pause_duration_ms)

    if not model_path.exists():
        if use_fallback:
            result = rule_based_prediction(utterance_text, pause_duration_ms)
            signals = extract_text_signals(utterance_text)
            return {
                "prediction": result["prediction"],
                "probability_end_of_turn": result["probability_end_of_turn"],
                "signals": signals,
                "baseline_700ms": int(pause_duration_ms >= 700),
                "model_type": "rule_based_fallback",
            }
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python src/train_classifier.py` first."
        )

    model = joblib.load(model_path)
    row = build_prediction_row(utterance_text, pause_duration_ms)
    prediction = int(model.predict(row)[0])

    if hasattr(model, "predict_proba"):
        probability_end = float(model.predict_proba(row)[0][1])
    else:
        probability_end = float(prediction)

    signals = extract_text_signals(utterance_text)
    return {
        "prediction": prediction,
        "probability_end_of_turn": probability_end,
        "signals": signals,
        "baseline_700ms": int(pause_duration_ms >= 700),
        "model_type": "trained_ml_classifier",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict end-of-turn for one utterance.")
    parser.add_argument("text")
    parser.add_argument("--pause-ms", type=int, default=700)
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="Fail if the trained model artifact is missing instead of using fallback rules.",
    )
    args = parser.parse_args()

    result = predict_end_of_turn(
        args.text,
        args.pause_ms,
        use_fallback=not args.require_model,
    )
    label = "End of Turn" if result["prediction"] else "Likely Continuing"
    print(f"{label} ({result['probability_end_of_turn']:.2%})")
    print(f"model_type={result['model_type']}")
    print(f"pause_baseline_700ms={result['baseline_700ms']}")


if __name__ == "__main__":
    main()
