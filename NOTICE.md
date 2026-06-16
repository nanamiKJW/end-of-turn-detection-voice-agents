# Notice and Data Attribution

This repository contains code, documentation, a synthetic demo dataset, and a small AMI-derived CSV for an end-of-turn detection portfolio project.

## AMI Meeting Corpus

The file `data/ami_turn_samples.csv` is derived from the AMI Meeting Corpus manual annotations.

- AMI Corpus: https://groups.inf.ed.ac.uk/ami/corpus/
- AMI download page: https://groups.inf.ed.ac.uk/ami/download/
- License: Creative Commons Attribution 4.0 International License, https://creativecommons.org/licenses/by/4.0/

The derived CSV was created by extracting word timing and speaker-change information from AMI annotation files, then assigning proxy end-of-turn labels.

This project is not an official AMI release and is not endorsed by the AMI Corpus creators.

## Project-Specific Notes

The AMI-derived labels are proxy labels:

- `1`: the next observed speech event is from another speaker
- `0`: the same speaker resumes before another speaker

These labels are useful for a portfolio/research prototype, but they should not be described as manually annotated voice-agent end-of-turn labels.
