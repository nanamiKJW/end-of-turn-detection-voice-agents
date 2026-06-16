"""Train a dependency-free text + metadata classifier.

This is a lightweight fallback for machines where scikit-learn cannot be
installed. It trains a simple Naive Bayes-style model using token counts and
Gaussian numeric features, then saves the model as JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "ami_turn_samples.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "lightweight_eot_model.json"

NUMERIC_FEATURES = [
    "pause_duration_ms",
    "speech_duration_ms",
    "num_words",
    "ends_with_punctuation",
    "has_question_word",
    "syntactic_completeness_score",
    "is_backchannel",
    "is_interruption",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?", text.lower())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def train(rows: list[dict[str, str]]) -> dict[str, object]:
    class_counts = Counter(row["label_end_of_turn"] for row in rows)
    token_counts = {"0": Counter(), "1": Counter()}
    token_totals = {"0": 0, "1": 0}
    numeric_values: dict[str, dict[str, list[float]]] = {
        "0": defaultdict(list),
        "1": defaultdict(list),
    }

    vocabulary = set()
    for row in rows:
        label = row["label_end_of_turn"]
        tokens = tokenize(row["utterance_text"])
        token_counts[label].update(tokens)
        token_totals[label] += len(tokens)
        vocabulary.update(tokens)

        for feature in NUMERIC_FEATURES:
            numeric_values[label][feature].append(float(row[feature]))

    numeric_stats = {"0": {}, "1": {}}
    for label in ("0", "1"):
        for feature in NUMERIC_FEATURES:
            values = numeric_values[label][feature]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
            numeric_stats[label][feature] = {
                "mean": mean,
                "std": max(math.sqrt(variance), 1e-6),
            }

    model = {
        "class_priors": {
            label: class_counts[label] / len(rows)
            for label in ("0", "1")
        },
        "token_counts": {
            label: dict(token_counts[label])
            for label in ("0", "1")
        },
        "token_totals": token_totals,
        "vocabulary_size": len(vocabulary),
        "numeric_stats": numeric_stats,
        "numeric_features": NUMERIC_FEATURES,
        "training_rows": len(rows),
        "model_type": "lightweight_trained_naive_bayes",
    }
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight end-of-turn model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    rows = read_rows(args.data)
    model = train(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Saved lightweight trained model to {args.output}")
    print(f"Training rows: {model['training_rows']}")


if __name__ == "__main__":
    main()
