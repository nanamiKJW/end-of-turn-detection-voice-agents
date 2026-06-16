"""Feature definitions for text-based end-of-turn classifiers."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TEXT_COLUMN = "utterance_text"
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


def build_feature_transformer() -> ColumnTransformer:
    """Create a transformer combining TF-IDF text features and metadata."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=1200,
                ),
                TEXT_COLUMN,
            ),
            ("metadata", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )
