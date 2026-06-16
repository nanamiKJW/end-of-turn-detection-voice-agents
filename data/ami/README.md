# AMI Meeting Corpus Data

This folder is for local AMI Meeting Corpus files. The raw AMI annotations are not committed to this repository.

## Why AMI?

The AMI Meeting Corpus is a real multi-speaker meeting corpus with around 100 hours of recordings, orthographic transcription, speaker/channel information, and several annotation layers. It is a better portfolio data source than a purely synthetic CSV because it contains natural spoken interaction, pauses, overlaps, disfluencies, and speaker changes.

Official source: https://groups.inf.ed.ac.uk/ami/corpus/

## Download

Download the manual annotations from the AMI download page:

https://groups.inf.ed.ac.uk/ami/download/

Use:

```text
AMI manual annotations v1.6.2
```

Unzip the annotation package under:

```text
data/ami/raw/
```

After unzipping, this project expects to find files ending in:

```text
*.words.xml
```

## Build the AMI-Derived Dataset

From the project root:

```bash
python src/prepare_ami_dataset.py --ami-root data/ami/raw --output data/ami_turn_samples.csv
```

This creates a real-corpus-derived CSV with the same main feature columns as the synthetic dataset, plus provenance columns such as `meeting_id`, `speaker_id`, `next_speaker_id`, `source_corpus`, and `label_source`.

## Labeling Method

The AMI-derived labels are proxy labels:

- `label_end_of_turn = 1`: after the candidate word, the next observed speech event is from a different speaker.
- `label_end_of_turn = 0`: after the candidate word, the same speaker resumes before another speaker.

This is useful for a portfolio and research-style prototype, but it is not the same as manually annotated voice-agent end-of-turn intent. Real deployment work would need task-specific labels, audio features, ASR uncertainty, and latency-aware evaluation.
