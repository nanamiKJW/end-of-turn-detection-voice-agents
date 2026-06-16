"""Train and compare text-plus-metadata classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from text_features import NUMERIC_FEATURES, TEXT_COLUMN, build_feature_transformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_turns.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "end_of_turn_classifier.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
LABEL_COLUMN = "label_end_of_turn"


MODEL_CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


def to_dense(matrix):
    """Convert sparse TF-IDF matrices to dense arrays for tree-based models."""

    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def split_data(data: pd.DataFrame, test_size: float = 0.25):
    x = data[[TEXT_COLUMN, *NUMERIC_FEATURES]]
    y = data[LABEL_COLUMN]
    return train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", build_feature_transformer()),
            ("to_dense", FunctionTransformer(to_dense, accept_sparse=True)),
            ("classifier", model),
        ]
    )


def train_and_compare(data: pd.DataFrame) -> tuple[str, Pipeline, pd.DataFrame]:
    x_train, x_test, y_train, y_test = split_data(data)
    rows = []
    trained_models: dict[str, Pipeline] = {}

    for model_name, model in MODEL_CANDIDATES.items():
        pipeline = build_pipeline(model)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )
        trained_models[model_name] = pipeline

    results = pd.DataFrame(rows).sort_values(["f1", "recall"], ascending=False)
    best_name = str(results.iloc[0]["model"])
    return best_name, trained_models[best_name], results


def save_artifacts(model: Pipeline, metrics: pd.DataFrame) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics.round(4).to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train end-of-turn classifiers.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    best_name, best_model, metrics = train_and_compare(data)
    save_artifacts(best_model, metrics)
    print(metrics.round(3).to_string(index=False))
    print(f"\nSaved best model: {best_name} -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
