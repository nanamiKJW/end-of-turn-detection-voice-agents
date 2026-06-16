"""Pause-threshold baseline for end-of-turn detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_turns.csv"


def predict_with_threshold(pause_duration_ms: pd.Series, threshold_ms: int) -> pd.Series:
    """Predict end-of-turn when the observed pause reaches a threshold."""

    return (pause_duration_ms >= threshold_ms).astype(int)


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute binary classification metrics without requiring scikit-learn."""

    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())

    total = true_positive + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_thresholds(
    data: pd.DataFrame,
    thresholds: tuple[int, ...] = (500, 700, 1000),
    label_col: str = "label_end_of_turn",
) -> pd.DataFrame:
    results = []
    y_true = data[label_col]

    for threshold in thresholds:
        y_pred = predict_with_threshold(data["pause_duration_ms"], threshold)
        metrics = binary_metrics(y_true, y_pred)
        results.append(
            {
                "threshold_ms": threshold,
                **metrics,
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pause-threshold baselines.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    print(evaluate_thresholds(data).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
