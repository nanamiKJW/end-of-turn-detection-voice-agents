# Data

`sample_turns.csv` is a synthetic dataset created for educational prototyping and portfolio demonstration. It is not collected from real conversations and should not be used to claim production-level performance.

For a more realistic experiment, the project also includes an AMI Meeting Corpus preparation pipeline. See `data/ami/README.md` and `src/prepare_ami_dataset.py`.

The examples are designed to represent common turn-taking cases:

- complete statements
- incomplete sentence fragments
- questions
- short backchannels such as "yes", "okay", and "right"
- hesitations such as "um", "well", and "I mean"
- interrupted or cut-off turns
- long, short, and ambiguous utterances

Columns include transcript text, pause duration, estimated speech duration, lightweight linguistic features, and the binary `label_end_of_turn` target.

## Labeling Assumptions

The label answers this question:

```text
Should a voice agent reasonably start responding now?
```

`label_end_of_turn = 1` means the turn is treated as complete in the synthetic scenario. `label_end_of_turn = 0` means the user is likely continuing, self-correcting, or was interrupted.

These labels are intentionally simplified. Real annotation would need clearer guidelines, multiple annotators, inter-annotator agreement, and access to audio context.

## Known Synthetic Biases

The dataset intentionally contains some paired examples, such as an incomplete version and a completed version of a similar utterance. This makes the contrast easy to inspect, but it can also make model evaluation easier than a real deployment setting.

The pause durations are plausible approximations, not measured audio. They are included to support baseline comparison and feature engineering.

## Real-Corpus Extension

The AMI-derived dataset is created locally as `data/ami_turn_samples.csv` after downloading the public AMI manual annotations. The derived CSV is small enough to include for portfolio reproducibility. The raw AMI annotation folder is ignored by Git.

The AMI labels are proxy labels based on speaker timing:

- next speaker changes -> end of turn
- same speaker resumes -> likely continuing

This makes the project more realistic than a synthetic-only demo, while keeping the limitations visible.

## Attribution

AMI Meeting Corpus:
https://groups.inf.ed.ac.uk/ami/corpus/

AMI download page:
https://groups.inf.ed.ac.uk/ami/download/

License:
Creative Commons Attribution 4.0 International License, https://creativecommons.org/licenses/by/4.0/

The AMI-derived CSV in this project is adapted from AMI manual annotations by deriving proxy labels from word timing and speaker changes. It is not an official AMI release.
