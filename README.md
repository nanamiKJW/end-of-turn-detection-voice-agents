# End-of-Turn Detection for Conversational Voice Agents

This project is a portfolio-style prototype for detecting whether a speaker has finished their turn in a spoken conversation. The task is useful for voice assistants, call-center agents, and conversational AI systems that need to decide when to respond without interrupting the user or waiting too long.

The project compares a simple pause-based baseline with a text-aware machine learning pipeline using transcript and metadata features. It includes a synthetic demo dataset and an AMI Meeting Corpus adapter for building real-corpus-derived examples from public meeting transcripts.

## What This Project Demonstrates

- Framing a speech/NLP problem as a measurable ML task
- Building a transparent baseline before training models
- Designing lightweight linguistic features from partial transcripts
- Comparing classical ML models with clear evaluation metrics
- Explaining product-relevant errors, not only aggregate scores
- Packaging the work as a small local demo and reproducible repository

## Motivation

Voice agents need good turn-taking. If the system responds too early, it interrupts the user. If it waits too long, the conversation feels slow. Silence alone is not always enough because speakers pause in the middle of a sentence, hesitate, self-correct, or give very short complete answers.

This prototype explores whether combining pause duration with linguistic cues can improve the decision:

```text
Input:  "I want to book a ticket to"
Output: Likely Continuing

Input:  "I want to book a ticket to Paris tomorrow."
Output: End of Turn
```

## Dataset

The project supports two dataset modes.

### Synthetic Prototype Data

The synthetic dataset is stored in `data/sample_turns.csv`. It contains 133 examples with a near-balanced label distribution. It was created for educational prototyping and quick demos. It is not a real conversational corpus and should not be used to claim production-level performance.

The examples cover:

- complete statements
- incomplete sentences
- questions
- backchannels such as "yes", "okay", and "right"
- hesitations such as "um", "well", and "I mean"
- interrupted turns
- long, short, and ambiguous utterances

Main columns:

- `utterance_text`
- `pause_duration_ms`
- `speech_duration_ms`
- `num_words`
- `last_token`
- `ends_with_punctuation`
- `has_question_word`
- `syntactic_completeness_score`
- `is_backchannel`
- `is_interruption`
- `label_end_of_turn`

The label answers a practical question: should a voice agent reasonably start responding now? In real annotation work, this would require clearer labeling guidelines, multiple annotators, and audio context.

### AMI Meeting Corpus Extension

For a more realistic version, the project includes a preparation script for the AMI Meeting Corpus:

```bash
python src/prepare_ami_dataset.py --ami-root data/ami/raw --output data/ami_turn_samples.csv
```

The AMI Meeting Corpus is a real multi-speaker meeting corpus with around 100 hours of recordings and public manual annotations. The AMI annotations include orthographic transcription and word-level timing information in NXT XML format.

The AMI-derived labels are proxy labels:

- `label_end_of_turn = 1`: the next observed speech event is from another speaker.
- `label_end_of_turn = 0`: the same speaker resumes before another speaker.

This makes the project more realistic, but the labels should still be described carefully. They are derived from speaker timing, not manually annotated voice-agent end-of-turn judgments.

The generated file `data/ami_turn_samples.csv` contains 6,000 AMI-derived examples balanced across the proxy labels. It is small enough to keep with the project, while the raw AMI annotation folder is ignored by Git.

## Methodology

### 1. Pause Baseline

The baseline predicts end-of-turn when:

```text
pause_duration_ms >= threshold
```

The project compares 500 ms, 700 ms, and 1000 ms thresholds. This baseline is transparent and useful, but limited because silence does not always mean the user is finished.

### 2. Text-Based ML Classifier

The main scikit-learn ML pipeline uses TF-IDF features from the utterance text and structured metadata features:

- number of words
- pause duration
- speech duration
- sentence-final punctuation
- question-word cue
- syntactic completeness score
- backchannel flag
- interruption flag

The training script compares Logistic Regression, Random Forest, and Gradient Boosting classifiers using accuracy, precision, recall, F1-score, and a confusion matrix.

Because the local development machine may not always have enough disk space for the full scikit-learn/Streamlit stack, the repository also includes a small dependency-free trained model:

```bash
python3 src/train_lightweight_classifier.py --data data/ami_turn_samples.csv
python3 app/local_demo.py
```

This lightweight model is trained on the AMI-derived CSV and is used by the local browser demo at `http://localhost:8501/`.

The model is intentionally classical rather than transformer-first. For this project, the main goal is to show a clean baseline, interpretable features, and careful evaluation on a small prototype dataset.

### 3. Advanced Modeling

A transformer-based classifier is left as future work. For a first portfolio version, the project stays deliberately lightweight so that the baseline, features, error analysis, and app are easy to inspect.

## Results

Baseline results on the synthetic dataset:

| Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 500 ms | 0.857 | 0.871 | 0.859 | 0.865 |
| 700 ms | 0.805 | 0.881 | 0.732 | 0.800 |
| 1000 ms | 0.534 | 1.000 | 0.127 | 0.225 |

The higher 1000 ms threshold is conservative: it avoids false end-of-turn predictions on this dataset but misses many completed turns. This mirrors the real design trade-off between politeness and responsiveness.

Baseline results on the AMI-derived dataset:

| Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 500 ms | 0.296 | 0.367 | 0.563 | 0.444 |
| 700 ms | 0.241 | 0.310 | 0.421 | 0.357 |
| 1000 ms | 0.197 | 0.238 | 0.276 | 0.256 |

These lower scores are expected and useful: real meeting data is messier than the synthetic examples, and speaker-change proxy labels are noisy. This is a better reflection of why end-of-turn detection is difficult.

Full scikit-learn classifier metrics are generated by:

```bash
python src/train_classifier.py
python src/evaluate.py
```

To train on AMI-derived examples after preparing the AMI CSV:

```bash
python src/train_classifier.py --data data/ami_turn_samples.csv
python src/evaluate.py --data data/ami_turn_samples.csv
```

The lightweight demo model is included as `models/lightweight_eot_model.json`. Larger generated scikit-learn artifacts are intentionally ignored by Git.

For this task, precision and recall have a product meaning:

- False end-of-turn predictions can make the assistant interrupt.
- False not-end-of-turn predictions can make the assistant wait too long.
- A practical system should choose a threshold based on the desired balance between responsiveness and politeness.

## Local Demo App

The repository includes two app options.

### Lightweight Local Demo

This is the easiest demo to run because it only needs Python:

```bash
python3 app/local_demo.py
```

Then open:

```text
http://localhost:8501/
```

The local demo uses `models/lightweight_eot_model.json`, a small trained model built from the AMI-derived CSV.

### Streamlit App

The Streamlit version is available in `app/streamlit_app.py` if the dependencies are installed. It includes:

- project introduction
- text input for a partial utterance
- pause duration slider
- end-of-turn prediction
- confidence score
- baseline vs ML comparison
- example utterances
- signal explanations
- model performance summary

If the scikit-learn artifact is missing, the Streamlit app can fall back to a transparent demo mode. For the current lightweight browser demo, `models/lightweight_eot_model.json` is already included.

Screenshot placeholders are in `screenshots/`.

## Run Locally

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/data_generation.py
python src/baseline_pause_detector.py
python src/error_analysis.py
python src/train_classifier.py
python src/evaluate.py
streamlit run app/streamlit_app.py
```

If you only want the working lightweight demo:

```bash
python3 src/train_lightweight_classifier.py --data data/ami_turn_samples.csv
python3 app/local_demo.py
```

Optional AMI extension after downloading and unzipping AMI manual annotations into `data/ami/raw/`:

```bash
python src/prepare_ami_dataset.py --ami-root data/ami/raw --output data/ami_turn_samples.csv
python src/train_classifier.py --data data/ami_turn_samples.csv
```

If the model file is missing, the Streamlit app will ask you to run the training script first.

## Project Structure

```text
end-of-turn-detection-voice-agents/
├── README.md
├── NOTICE.md
├── requirements.txt
├── data/
│   ├── ami/
│   │   └── README.md
│   ├── ami_turn_samples.csv
│   ├── sample_turns.csv
│   └── README.md
├── notebooks/
│   ├── 01_problem_exploration.ipynb
│   ├── 02_baseline_pause_detection.ipynb
│   ├── 03_text_classifier_training.ipynb
│   └── 04_evaluation_analysis.ipynb
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── baseline_pause_detector.py
│   ├── error_analysis.py
│   ├── prepare_ami_dataset.py
│   ├── text_features.py
│   ├── train_lightweight_classifier.py
│   ├── train_classifier.py
│   ├── evaluate.py
│   └── predict.py
├── app/
│   ├── local_demo.py
│   └── streamlit_app.py
├── models/
│   ├── lightweight_eot_model.json
│   └── README.md
├── reports/
│   ├── baseline_error_analysis.md
│   ├── model_report.md
│   ├── recruiter_review_checklist.md
│   └── research_summary.md
└── screenshots/
    └── README.md
```

## Limitations

This is a prototype, not a production system. The synthetic dataset is simplified, and the text completeness score is a heuristic. Some synthetic examples are paired incomplete/complete variants, which is useful for demonstration but easier than a natural test set. The AMI-derived data is more realistic, but its labels are still proxy labels based on next-speaker timing. Real voice-agent systems would need task-specific annotation, audio features, ASR uncertainty, prosody, dialogue context, and latency-aware evaluation.

## Future Improvements

- Add manually reviewed turn-completion labels for a subset of AMI-derived examples.
- Include prosodic features such as pitch, energy, and final lengthening.
- Evaluate streaming ASR partial transcripts.
- Add multilingual or low-resource language examples.
- Try sentence embeddings or a small transformer model.
- Calibrate the decision threshold for different assistant behaviors.
- Use grouped train/test splits or external test data to reduce near-duplicate leakage.
- Add punctuation-free ASR-style transcripts because real streaming ASR often does not provide perfect punctuation.

## References

- AMI Meeting Corpus: https://groups.inf.ed.ac.uk/ami/corpus/
- AMI Corpus download page: https://groups.inf.ed.ac.uk/ami/download/
- AMI transcription documentation: https://groups.inf.ed.ac.uk/ami/corpus/transcription.shtml

## Data Attribution

This project uses derived examples from the AMI Meeting Corpus manual annotations.

The AMI corpus transcription and annotations are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

This repository derives proxy end-of-turn labels from AMI word timing and speaker changes. The derived dataset and models are not official AMI resources and are not endorsed by the AMI Corpus creators.

## What I Learned

This project connects linguistic analysis with applied ML engineering. It shows how a practical conversational AI problem can be approached with a transparent baseline, interpretable features, model comparison, and a small interactive demo. It also highlights that the right evaluation question is not only "Which model has the best F1?" but "What kind of conversational error does this system make?"
