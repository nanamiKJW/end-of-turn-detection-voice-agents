"""Error analysis utilities for the pause-threshold baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline_pause_detector import predict_with_threshold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_turns.csv"


def baseline_errors(data: pd.DataFrame, threshold_ms: int = 700) -> pd.DataFrame:
    """Return examples where a pause-only baseline disagrees with the label."""

    analysis = data.copy()
    analysis["baseline_prediction"] = predict_with_threshold(
        analysis["pause_duration_ms"],
        threshold_ms,
    )
    analysis["error_type"] = "correct"
    analysis.loc[
        (analysis["label_end_of_turn"] == 0) & (analysis["baseline_prediction"] == 1),
        "error_type",
    ] = "false_end_of_turn"
    analysis.loc[
        (analysis["label_end_of_turn"] == 1) & (analysis["baseline_prediction"] == 0),
        "error_type",
    ] = "false_not_end_of_turn"

    columns = [
        "turn_id",
        "utterance_text",
        "pause_duration_ms",
        "syntactic_completeness_score",
        "label_end_of_turn",
        "baseline_prediction",
        "error_type",
    ]
    return analysis.loc[analysis["error_type"] != "correct", columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect pause-baseline errors.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--threshold-ms", type=int, default=700)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    errors = baseline_errors(data, args.threshold_ms)
    print(f"Baseline threshold: {args.threshold_ms} ms")
    print(errors["error_type"].value_counts().to_string())
    print()
    print(errors.head(args.limit).to_string(index=False))


if __name__ == "__main__":
    main()
