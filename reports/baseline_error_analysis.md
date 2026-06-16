# Baseline Error Analysis

This report explains where the 700 ms pause baseline is expected to fail. The goal is not to make the baseline look bad; it is to show why end-of-turn detection needs more than silence duration.

Run:

```bash
python src/error_analysis.py --threshold-ms 700
```

## Main Failure Types

False end-of-turn errors happen when the user pauses for a relatively long time but the utterance is still incomplete.

Examples:

- "The reason I called is because"
- "Could you help me with"
- "I need to travel on"

In a live voice agent, these errors are risky because the assistant may interrupt while the user is still formulating the request.

False not-end-of-turn errors happen when the user gives a short complete answer but the pause is below the baseline threshold.

Examples:

- "Yes."
- "No thanks."
- "By email."
- "Two people."

These errors make the assistant wait unnecessarily even though the user has already provided a complete response.

## Why This Matters

The baseline is still valuable because it gives a simple reference point. However, the failure modes show why transcript features are useful:

- incomplete final tokens such as "to", "with", or "because"
- punctuation or question form
- backchannels and short answers
- interruption markers
- semantic or syntactic completeness cues

For a real system, the same analysis should be repeated with natural ASR transcripts and audio-derived pause/prosody features.
