"""Streamlit demo for end-of-turn detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline_pause_detector import evaluate_thresholds  # noqa: E402
from predict import LIGHTWEIGHT_MODEL_PATH, MODEL_PATH, predict_end_of_turn  # noqa: E402
from preprocessing import extract_text_signals  # noqa: E402
from rule_based_detector import rule_based_prediction  # noqa: E402


SYNTHETIC_DATA_PATH = PROJECT_ROOT / "data" / "sample_turns.csv"
AMI_DATA_PATH = PROJECT_ROOT / "data" / "ami_turn_samples.csv"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

EXAMPLES = [
    "I want to book a flight to",
    "I want to book a flight to Paris.",
    "Can you help me with my visa application?",
    "Because I was thinking that maybe",
    "The reason I called is because",
    "Yeah.",
    "Okay, thanks.",
    "Wait, I mean",
]


st.set_page_config(
    page_title="End-of-Turn Detection",
    page_icon=":microphone:",
    layout="wide",
)


def available_datasets() -> dict[str, Path]:
    datasets = {}
    if SYNTHETIC_DATA_PATH.exists():
        datasets["Synthetic prototype data"] = SYNTHETIC_DATA_PATH
    if AMI_DATA_PATH.exists():
        datasets["AMI-derived real-corpus data"] = AMI_DATA_PATH
    return datasets


def load_data(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def load_metrics() -> list[dict[str, object]]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return []


def signal_explanations(text: str, pause_ms: int) -> list[str]:
    signals = extract_text_signals(text)
    notes = []
    if pause_ms >= 700:
        notes.append("Pause is long enough for the 700 ms baseline to respond.")
    else:
        notes.append("Pause is short, so a silence-only system would likely keep waiting.")
    if signals.ends_with_punctuation:
        notes.append("The transcript has sentence-final punctuation.")
    if signals.has_question_word:
        notes.append("The utterance looks like a question or request.")
    if signals.is_backchannel:
        notes.append("The text is a short backchannel such as yes, okay, or right.")
    if signals.is_interruption:
        notes.append("The transcript contains an interruption or cut-off marker.")
    if signals.syntactic_completeness_score < 0.45:
        notes.append("The lightweight completeness score suggests the sentence is unfinished.")
    return notes


st.title("End-of-Turn Detection for Conversational Voice Agents")
st.caption("A portfolio prototype comparing pause-based and text-aware turn completion signals.")

intro_col, metric_col = st.columns([1.6, 1])
with intro_col:
    st.write(
        "Voice agents need to decide when a user has finished speaking. Responding too early "
        "interrupts the user; waiting too long makes the assistant feel slow. This demo combines "
        "pause duration, transcript text, and small linguistic metadata features."
    )
with metric_col:
    dataset_options = available_datasets()
    selected_dataset = st.selectbox(
        "Dataset for summary tables",
        list(dataset_options.keys()) or ["No dataset found"],
    )
    selected_data_path = dataset_options.get(selected_dataset)
    data = load_data(selected_data_path) if selected_data_path else None
    if data is not None:
        st.metric("Dataset examples", len(data))
        st.metric("End-of-turn rate", f"{data['label_end_of_turn'].mean():.0%}")
        if LIGHTWEIGHT_MODEL_PATH.exists():
            st.metric("Demo model", "Trained lightweight")
        elif MODEL_PATH.exists():
            st.metric("Demo model", "scikit-learn")
        else:
            st.metric("Demo model", "Fallback")
    else:
        st.warning("Dataset not found. Run `python src/data_generation.py`.")

st.divider()

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Try an utterance")
    selected = st.selectbox("Example utterances", EXAMPLES)
    utterance = st.text_area("Partial transcript", value=selected, height=110)
    pause_ms = st.slider("Observed pause after speech (ms)", 100, 1800, 700, step=50)
    decision_threshold = st.slider(
        "Decision threshold",
        0.10,
        0.90,
        0.50,
        step=0.05,
        help="Higher thresholds make the assistant more conservative about responding.",
    )
    run_prediction = st.button("Predict", type="primary")

with right:
    st.subheader("Prediction")
    if run_prediction:
        if not utterance.strip():
            st.error("Please enter an utterance.")
        else:
            try:
                has_trained_model = MODEL_PATH.exists() or LIGHTWEIGHT_MODEL_PATH.exists()
                if has_trained_model:
                    result = predict_end_of_turn(utterance, pause_ms)
                    model_type = str(result.get("model_type", "trained_model"))
                    if model_type == "lightweight_trained_model":
                        model_name = "Trained lightweight model"
                    elif model_type == "trained_ml_classifier":
                        model_name = "scikit-learn classifier"
                    else:
                        model_name = "Trained model"
                else:
                    result = rule_based_prediction(
                        utterance,
                        pause_ms,
                        threshold=decision_threshold,
                    )
                    model_name = "Rule-based fallback"

                probability = float(result["probability_end_of_turn"])
                is_end = probability >= decision_threshold
                st.metric(
                    model_name,
                    "End of Turn" if is_end else "Likely Continuing",
                    f"{probability:.1%} probability",
                )
                baseline_label = "End of Turn" if pause_ms >= 700 else "Likely Continuing"
                st.metric("Pause baseline at 700 ms", baseline_label)

                if not has_trained_model:
                    st.warning(
                        "Trained model not found. Showing a transparent rule-based "
                        "fallback. Run `python src/train_lightweight_classifier.py "
                        "--data data/ami_turn_samples.csv` for the lightweight model."
                    )

                st.write("Important signals")
                for note in signal_explanations(utterance, pause_ms):
                    st.write(f"- {note}")
            except Exception as exc:  # pragma: no cover - Streamlit display path
                st.error(f"Prediction failed: {exc}")
    else:
        st.info("Enter an utterance and click Predict.")

st.divider()

baseline_col, model_col = st.columns(2)
with baseline_col:
    st.subheader("Pause baseline")
    if data is not None:
        st.dataframe(evaluate_thresholds(data).round(3), use_container_width=True)
    else:
        st.info("Generate the dataset to view baseline results.")

with model_col:
    st.subheader("ML model comparison")
    metrics = load_metrics()
    if metrics:
        st.dataframe(pd.DataFrame(metrics).round(3), use_container_width=True)
    else:
        st.info("Train the classifier to view model metrics.")

st.divider()
st.subheader("Interpretation")
st.write(
    "For voice agents, false end-of-turn predictions can cause interruptions, while false "
    "not-end-of-turn predictions add latency. In practice, the decision threshold should be "
    "chosen based on product goals: faster responses, more polite turn-taking, or a balance of both."
)
