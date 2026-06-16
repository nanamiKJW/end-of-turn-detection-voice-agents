# Model Report

## Dataset

The project uses `data/sample_turns.csv`, a synthetic dataset with 133 examples. It includes complete utterances, incomplete fragments, questions, backchannels, hesitations, interrupted turns, short turns, long turns, and ambiguous cases.

The dataset is for educational prototyping and portfolio demonstration. It should not be interpreted as evidence of real-world production performance.

The project now also includes an AMI Meeting Corpus preparation script. After downloading AMI manual annotations, run:

```bash
python src/prepare_ami_dataset.py --ami-root data/ami/raw --output data/ami_turn_samples.csv
```

The AMI-derived CSV can be used with the same baseline, training, and evaluation scripts by passing `--data data/ami_turn_samples.csv`.

The current AMI-derived file contains 6,000 examples:

- Proxy end of turn: 3,000
- Proxy likely continuing: 3,000

Label balance:

- End of turn: 71 examples
- Likely continuing: 62 examples

The synthetic data deliberately includes difficult baseline cases: incomplete utterances with relatively long pauses and short complete answers with relatively short pauses.

## Baseline Results

The pause-based baseline was run locally on the full synthetic dataset.

| Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 500 ms | 0.857 | 0.871 | 0.859 | 0.865 |
| 700 ms | 0.805 | 0.881 | 0.732 | 0.800 |
| 1000 ms | 0.534 | 1.000 | 0.127 | 0.225 |

The 500 ms threshold performs best on this synthetic dataset, but the result should be read carefully. A lower threshold is more responsive and catches more completed turns, while a higher threshold is more conservative and reduces interruption risk.

AMI-derived baseline results:

| Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 500 ms | 0.296 | 0.367 | 0.563 | 0.444 |
| 700 ms | 0.241 | 0.310 | 0.421 | 0.357 |
| 1000 ms | 0.197 | 0.238 | 0.276 | 0.256 |

The AMI numbers are much lower because natural meetings include overlaps, backchannels, short acknowledgements, long pauses, and noisy proxy labels. This makes the real-data extension more convincing than a synthetic-only benchmark.

This is exactly why the task is interesting. A silence threshold is a useful engineering baseline, but it cannot tell whether "The reason I called is because" is incomplete or whether "Okay, thanks" is already a complete turn.

## ML Classifier

The full scikit-learn training pipeline compares:

- Logistic Regression
- Random Forest
- Gradient Boosting

Features include TF-IDF text n-grams plus metadata such as pause duration, speech duration, word count, punctuation, question-word cues, a lightweight completeness score, backchannel status, and interruption status.

Run:

```bash
python src/train_classifier.py
python src/evaluate.py
```

The training script saves:

- `models/end_of_turn_classifier.joblib`
- `models/metrics.json`

These files are ignored by Git so the project stays lightweight and reproducible.

## Lightweight Demo Model

For machines where the full scikit-learn/Streamlit environment is too heavy, the project includes a dependency-free trained model:

```bash
python3 src/train_lightweight_classifier.py --data data/ami_turn_samples.csv
python3 app/local_demo.py
```

This model is saved as `models/lightweight_eot_model.json` and is used by the local browser demo. It is a simple Naive Bayes-style classifier using word counts and numeric metadata features. It is included for a working portfolio demo, not as a claim of state-of-the-art performance.

## Metric Interpretation

For a voice agent, the most important metric depends on the product goal:

- High precision for end-of-turn reduces interruptions.
- High recall for end-of-turn reduces awkward waiting.
- F1-score is useful for comparison, but it hides the product trade-off.

For internship discussion, the strongest point is not the absolute score on synthetic data. It is the modeling setup: a clear baseline, interpretable features, multiple classifiers, and an explicit discussion of error costs.

## Error Analysis Plan

Before presenting final ML metrics, inspect:

- false end-of-turn predictions, because these correspond to interruptions
- false not-end-of-turn predictions, because these correspond to delayed responses
- disagreements between the pause baseline and the ML classifier
- examples with short complete answers and long incomplete pauses

The local demo uses the lightweight trained model when `models/lightweight_eot_model.json` exists. If that file is missing, it can still fall back to transparent rules, but the included GitHub version should contain the trained lightweight model.
