# Research Summary: End-of-Turn Detection for Conversational Voice Agents

## Problem

End-of-turn detection asks whether a speaker has finished their current turn or is likely to continue. In a voice agent, this decision affects when the system should start responding.

This is a small but important conversational AI problem. If the assistant responds too early, it interrupts the user. If it waits too long, the interaction feels slow or unnatural. A pause-only strategy is easy to implement, but silence is not always enough: some users pause mid-sentence, while short answers such as "yes" or "okay" may be complete even with short speech duration.

## Why It Matters

Turn-taking is part of the user experience of speech systems. End-of-turn detection connects speech processing, NLP, dialogue systems, and latency-aware engineering. A practical system may combine several signals:

- audio pause duration
- partial ASR transcript text
- prosody and intonation
- semantic or syntactic completeness
- dialogue state and task context
- user-specific speaking style

This prototype focuses on pause duration and transcript-based linguistic cues. It starts with a synthetic dataset for controlled examples and includes an AMI Meeting Corpus pipeline for real-corpus-derived experiments.

## Baseline Method

The baseline predicts end-of-turn when:

```text
pause_duration_ms >= threshold
```

The project evaluates thresholds of 500 ms, 700 ms, and 1000 ms. This baseline is useful because it is transparent and easy to deploy, but it cannot distinguish between a complete short answer and a mid-sentence pause.

## Machine Learning Method

The text-based ML approach uses:

- TF-IDF features from `utterance_text`
- pause duration
- speech duration
- number of words
- punctuation cue
- question-word cue
- lightweight syntactic completeness score
- backchannel flag
- interruption flag

The training script compares Logistic Regression, Random Forest, and Gradient Boosting classifiers. The best model is saved as a joblib artifact for the Streamlit app.

For the AMI extension, labels are derived from word timing and speaker change. This is a realistic proxy for turn transition behavior, but it is not the same as manually annotated voice-agent readiness.

## Evaluation

Evaluation uses accuracy, precision, recall, F1-score, and a confusion matrix. For this task, the error types matter more than a single score:

- False end-of-turn: the assistant may interrupt the user.
- False not-end-of-turn: the assistant may wait too long.

In many conversational products, false end-of-turn errors are especially costly because interruptions feel impolite. However, very conservative systems can become frustrating if they add too much latency.

The prototype therefore treats evaluation as a trade-off analysis rather than a race for the highest score. A recruiter or researcher should read the baseline table together with error examples, especially cases where a user pauses mid-sentence or gives a short complete answer.

## Limitations

The dataset is synthetic and intended for prototyping, not for real deployment claims. The completeness score is a transparent heuristic, not a syntactic parser. The model does not use raw audio, prosody, speaker identity, ASR uncertainty, or dialogue history.

Because the examples are generated with clear labels, evaluation results may be optimistic compared with real conversational data.

Another limitation is that some synthetic examples are paired templates, for example an incomplete phrase and a completed version of the same phrase. This is useful for demonstrating the task, but a real test set should avoid near-duplicate leakage and should include natural ASR errors.

The AMI-derived data reduces the synthetic-only weakness, but proxy labeling still has noise. A speaker change can happen after an interruption, overlap, or backchannel, and a same-speaker continuation does not always mean the user intended to continue.

## Future Work

Useful extensions would include:

- collecting or adapting real dialogue data with turn-completion annotations
- adding audio features such as pitch, energy, final lengthening, and silence shape
- using streaming ASR partial transcripts
- evaluating by latency and interruption rate, not only classification metrics
- testing cross-lingual or low-resource settings
- adding a small transformer or sentence-embedding model
- calibrating decision thresholds for different assistant personalities or tasks
