"""Streamlit portfolio demo for end-of-turn detection."""

from __future__ import annotations

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

EXAMPLES = {
    "Incomplete request": "I want to book a flight to",
    "Complete request": "I want to book a flight to Paris tomorrow.",
    "Question": "Can you help me with my visa application?",
    "Mid-sentence pause": "The reason I called is because",
    "Short answer": "Okay, thanks.",
    "Repair / hesitation": "Wait, I mean",
}

RESPONSE_STYLES = {
    "Balanced": 0.50,
    "Respond earlier": 0.35,
    "Wait longer": 0.65,
}


st.set_page_config(
    page_title="End-of-Turn Detection",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .hero {
        border-bottom: 1px solid #dde3ea;
        padding-bottom: 1.1rem;
        margin-bottom: 1.4rem;
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 760;
        margin-bottom: .3rem;
        color: #1f2933;
    }
    .hero-copy {
        color: #53606f;
        font-size: 1.02rem;
        line-height: 1.55;
        max-width: 840px;
    }
    .result-box {
        border: 1px solid #d9e1e8;
        border-radius: 8px;
        padding: 1.05rem 1.1rem;
        background: #ffffff;
        margin-bottom: 1rem;
    }
    .result-label {
        font-size: 1.65rem;
        font-weight: 780;
        color: #1f2933;
        margin-bottom: .25rem;
    }
    .result-copy {
        color: #4b5563;
        line-height: 1.5;
    }
    .note {
        border-left: 4px solid #2f6f8f;
        background: #f5f8fa;
        padding: .85rem 1rem;
        border-radius: 6px;
        color: #334155;
        line-height: 1.5;
    }
    .small-muted {
        color: #64748b;
        font-size: .9rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dfe5ec;
        border-radius: 8px;
        padding: .8rem .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def available_datasets() -> dict[str, Path]:
    datasets = {}
    if AMI_DATA_PATH.exists():
        datasets["AMI-derived real-corpus data"] = AMI_DATA_PATH
    if SYNTHETIC_DATA_PATH.exists():
        datasets["Synthetic prototype data"] = SYNTHETIC_DATA_PATH
    return datasets


def get_model_name() -> str:
    if LIGHTWEIGHT_MODEL_PATH.exists():
        return "AMI timing classifier"
    if MODEL_PATH.exists():
        return "scikit-learn classifier"
    return "fallback scorer"


def predict_for_ui(utterance: str, pause_ms: int, threshold: float) -> dict[str, object]:
    has_trained_model = MODEL_PATH.exists() or LIGHTWEIGHT_MODEL_PATH.exists()
    if has_trained_model:
        result = predict_end_of_turn(utterance, pause_ms)
    else:
        result = rule_based_prediction(utterance, pause_ms, threshold=threshold)

    probability = float(result["probability_end_of_turn"])
    is_end = probability >= threshold
    signals = extract_text_signals(utterance)

    if is_end:
        action = "Respond"
        explanation = "The current turn looks complete enough for a response."
    else:
        action = "Wait"
        explanation = "The current turn still looks unfinished."

    return {
        "action": action,
        "explanation": explanation,
        "probability": probability,
        "is_end": is_end,
        "signals": signals,
        "baseline_label": "End of turn" if pause_ms >= 700 else "Likely continuing",
        "model_type": result.get("model_type", "fallback"),
    }


def signal_notes(utterance: str, pause_ms: int) -> list[str]:
    signals = extract_text_signals(utterance)
    notes = []
    if pause_ms >= 700:
        notes.append("Pause is above the 700 ms baseline threshold.")
    else:
        notes.append("Pause is below the 700 ms baseline threshold.")
    if signals.last_token:
        notes.append(f"Last token: `{signals.last_token}`.")
    if signals.last_token in {"to", "and", "or", "because", "with", "about", "for"}:
        notes.append("This final token often leaves a phrase open.")
    if signals.ends_with_punctuation:
        notes.append("Sentence-final punctuation is present.")
    if signals.is_backchannel:
        notes.append("This looks like a short response or backchannel.")
    notes.append(f"Completeness score: `{signals.syntactic_completeness_score}`.")
    return notes


st.markdown(
    """
    <div class="hero">
      <div class="hero-title">End-of-Turn Detection for Voice Agents</div>
      <div class="hero-copy">
        A small experiment on deciding whether a spoken turn is complete.
        It compares a pause-only baseline with a transcript-based classifier
        built from AMI-derived meeting examples.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

datasets = available_datasets()
if not datasets:
    st.error("No dataset found. The repository should include data/ami_turn_samples.csv.")
    st.stop()

selected_dataset_name = st.sidebar.selectbox("Dataset", list(datasets.keys()))
selected_path = datasets[selected_dataset_name]
data = load_dataset(str(selected_path))

st.sidebar.markdown("### Classifier")
st.sidebar.write(get_model_name())
st.sidebar.markdown("### Decision setting")
style_name = st.sidebar.radio(
    "Choose the threshold behavior",
    list(RESPONSE_STYLES.keys()),
    index=0,
)
decision_threshold = RESPONSE_STYLES[style_name]
st.sidebar.caption(
    "Lower thresholds respond sooner. Higher thresholds wait for stronger evidence."
)

top_metrics = st.columns(4)
top_metrics[0].metric("Examples", f"{len(data):,}")
top_metrics[1].metric("End-turn labels", f"{data['label_end_of_turn'].mean():.0%}")
top_metrics[2].metric("Classifier", get_model_name())
top_metrics[3].metric("Decision threshold", f"{decision_threshold:.0%}")

demo_tab, data_tab, notes_tab = st.tabs(["Demo", "Data and Baseline", "Notes"])

with demo_tab:
    input_col, result_col = st.columns([1.1, 1], gap="large")

    with input_col:
        st.subheader("Input")
        example_name = st.selectbox("Example", list(EXAMPLES.keys()))
        utterance = st.text_area(
            "Transcript so far",
            value=EXAMPLES[example_name],
            height=120,
            help="Type a partial utterance or choose one of the examples.",
        )
        pause_ms = st.slider(
            "Pause after speech",
            min_value=100,
            max_value=1800,
            value=400 if example_name == "Incomplete request" else 700,
            step=50,
            format="%d ms",
        )

        st.markdown(
            """
            <div class="note">
            Silence is useful, but not sufficient. Short answers can be complete,
            and long pauses can still occur inside an unfinished sentence.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col:
        st.subheader("Output")
        prediction = predict_for_ui(utterance, pause_ms, decision_threshold)

        st.markdown(
            f"""
            <div class="result-box">
              <div class="result-label">{prediction['action']}</div>
              <div class="result-copy">{prediction['explanation']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        score_cols = st.columns(2)
        score_cols[0].metric("Classifier score", f"{prediction['probability']:.1%}")
        score_cols[1].metric("Pause-only baseline", prediction["baseline_label"])

        st.markdown("#### Signals")
        for note in signal_notes(utterance, pause_ms):
            st.write(f"- {note}")

        with st.expander("Feature values"):
            signals = prediction["signals"]
            st.json(
                {
                    "num_words": signals.num_words,
                    "last_token": signals.last_token,
                    "ends_with_punctuation": signals.ends_with_punctuation,
                    "has_question_word": signals.has_question_word,
                    "syntactic_completeness_score": signals.syntactic_completeness_score,
                    "is_backchannel": signals.is_backchannel,
                    "is_interruption": signals.is_interruption,
                }
            )

with data_tab:
    baseline = evaluate_thresholds(data).round(3)
    st.subheader("Pause Baseline")
    st.write(
        "The baseline uses only pause duration. It is a useful reference point, "
        "but real turn-taking is not only a silence-threshold problem."
    )
    st.dataframe(baseline, use_container_width=True, hide_index=True)

    chart_data = baseline.set_index("threshold_ms")[["precision", "recall", "f1"]]
    st.bar_chart(chart_data)

    st.subheader("Dataset Preview")
    preview_columns = [
        "utterance_text",
        "pause_duration_ms",
        "speech_duration_ms",
        "label_end_of_turn",
    ]
    st.dataframe(data[preview_columns].head(12), use_container_width=True, hide_index=True)

with notes_tab:
    st.subheader("How to Read the Experiment")
    st.write(
        "This is a prototype, not a production voice-agent system. "
        "The AMI-derived labels are proxy labels based on speaker timing: a speaker change is treated "
        "as an end-of-turn signal, while same-speaker continuation is treated as likely continuing."
    )
    st.write(
        "The useful part is the workflow: start with a baseline, build a real-derived dataset, "
        "inspect interpretable features, and discuss interruption versus latency."
    )
    st.markdown("#### Error Trade-Off")
    st.write("- False end-of-turn: the assistant may interrupt the user.")
    st.write("- False not-end-of-turn: the assistant may wait too long.")
    st.markdown("#### Future Work")
    st.write("- Add audio features such as pitch, energy, and final lengthening.")
    st.write("- Evaluate streaming ASR transcripts without perfect punctuation.")
    st.write("- Manually review a subset of AMI-derived labels.")
