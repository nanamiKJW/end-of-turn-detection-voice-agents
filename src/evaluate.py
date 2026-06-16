"""Evaluation helpers for model and baseline comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline_pause_detector import evaluate_thresholds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_turns.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "end_of_turn_classifier.joblib"
LABEL_COLUMN = "label_end_of_turn"


def evaluate_saved_model(data: pd.DataFrame, model_path: Path = MODEL_PATH) -> dict[str, object]:
    import joblib
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    from text_features import NUMERIC_FEATURES, TEXT_COLUMN

    model = joblib.load(model_path)
    x = data[[TEXT_COLUMN, *NUMERIC_FEATURES]]
    y = data[LABEL_COLUMN]
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    predictions = model.predict(x_test)
    return {
        "classification_report": classification_report(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline and saved ML model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    print("Pause baseline:")
    print(evaluate_thresholds(data).round(3).to_string(index=False))

    if not args.model.exists():
        print(
            "\nSaved classifier not found. Run `python src/train_classifier.py` "
            "to generate model metrics."
        )
        return

    print("\nSaved classifier:")
    result = evaluate_saved_model(data, args.model)
    print(result["classification_report"])
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(result["confusion_matrix"])


if __name__ == "__main__":
    main()
