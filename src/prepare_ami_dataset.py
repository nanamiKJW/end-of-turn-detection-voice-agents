"""Prepare AMI Meeting Corpus word annotations for end-of-turn experiments.

The AMI annotations are distributed as NXT XML files. This script expects the
manual annotation ZIP to be downloaded and unzipped locally, then derives proxy
end-of-turn labels from word timing and speaker change.

Label definition:
- 1: after the current word, the next observed speech event is from another speaker
- 0: after the current word, the same speaker resumes before another speaker

This is a practical proxy label, not a human end-of-turn annotation.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from preprocessing import extract_text_signals, get_last_token


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AMI_ROOT = PROJECT_ROOT / "data" / "ami" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ami_turn_samples.csv"


WORD_FILE_RE = re.compile(
    r"(?P<meeting>[A-Z]{2}\d{4}[a-d]?)\.(?P<speaker>[A-Z])\.words\.xml$"
)
END_PUNCTUATION = {".", "?", "!", "-"}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def clean_word(text: str | None) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = text.replace("$", "").replace("%", "").replace("#", "")
    return re.sub(r"\s+", " ", text)


def parse_word_file(path: Path) -> list[dict[str, object]]:
    match = WORD_FILE_RE.search(path.name)
    if not match:
        return []

    meeting_id = match.group("meeting")
    speaker_id = match.group("speaker")
    root = ET.parse(path).getroot()
    words = []

    for element in root.iter():
        if local_name(element.tag) != "w":
            continue
        token = clean_word(element.text)
        if not token:
            continue
        try:
            start = float(element.attrib["starttime"])
            end = float(element.attrib["endtime"])
        except (KeyError, ValueError):
            continue
        if end < start:
            continue
        words.append(
            {
                "meeting_id": meeting_id,
                "speaker_id": speaker_id,
                "word": token,
                "start": start,
                "end": end,
                "source_file": str(path),
            }
        )

    return words


def load_ami_words(ami_root: Path) -> list[dict[str, object]]:
    word_files = sorted(ami_root.rglob("*.words.xml"))
    if not word_files:
        raise FileNotFoundError(
            f"No AMI .words.xml files found under {ami_root}. "
            "Download and unzip AMI manual annotations first."
        )

    events = []
    for path in word_files:
        events.extend(parse_word_file(path))
    return sorted(events, key=lambda item: (item["meeting_id"], item["start"], item["end"]))


def speaker_histories(events: list[dict[str, object]]) -> dict[tuple[str, str], list[int]]:
    histories: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        key = (str(event["meeting_id"]), str(event["speaker_id"]))
        histories[key].append(index)
    return histories


def build_context(
    events: list[dict[str, object]],
    speaker_indices: list[int],
    position: int,
    context_words: int,
    reset_gap_s: float,
) -> tuple[str, float]:
    selected = []
    current_event = events[position]
    previous_end = float(current_event["end"])

    for idx in reversed(speaker_indices):
        if idx > position:
            continue
        event = events[idx]
        gap = previous_end - float(event["end"])
        if selected and gap > reset_gap_s:
            break
        selected.append(event)
        previous_end = float(event["start"])
        if len(selected) >= context_words:
            break

    selected = list(reversed(selected))
    text = " ".join(str(event["word"]) for event in selected)
    speech_duration_ms = max(
        1,
        int((float(selected[-1]["end"]) - float(selected[0]["start"])) * 1000),
    )
    return text, speech_duration_ms


def derive_rows(
    events: list[dict[str, object]],
    min_pause_ms: int = 200,
    max_pause_ms: int = 3000,
    context_words: int = 18,
    reset_gap_s: float = 2.5,
    max_rows_per_label: int | None = 3000,
) -> list[dict[str, object]]:
    histories = speaker_histories(events)
    rows = []
    counts = {0: 0, 1: 0}

    for index, event in enumerate(events[:-1]):
        next_event = events[index + 1]
        if event["meeting_id"] != next_event["meeting_id"]:
            continue

        pause_ms = int(max(0.0, float(next_event["start"]) - float(event["end"])) * 1000)
        if pause_ms < min_pause_ms or pause_ms > max_pause_ms:
            continue

        label = int(event["speaker_id"] != next_event["speaker_id"])
        if max_rows_per_label is not None and counts[label] >= max_rows_per_label:
            continue

        key = (str(event["meeting_id"]), str(event["speaker_id"]))
        text, speech_duration_ms = build_context(
            events,
            histories[key],
            index,
            context_words=context_words,
            reset_gap_s=reset_gap_s,
        )
        signals = extract_text_signals(text)
        if signals.num_words < 2:
            continue

        counts[label] += 1
        rows.append(
            {
                "turn_id": f"ami_{len(rows) + 1:06d}",
                "meeting_id": event["meeting_id"],
                "speaker_id": event["speaker_id"],
                "next_speaker_id": next_event["speaker_id"],
                "utterance_text": text,
                "pause_duration_ms": pause_ms,
                "speech_duration_ms": speech_duration_ms,
                "num_words": signals.num_words,
                "last_token": get_last_token(text),
                "ends_with_punctuation": signals.ends_with_punctuation,
                "has_question_word": signals.has_question_word,
                "syntactic_completeness_score": signals.syntactic_completeness_score,
                "is_backchannel": signals.is_backchannel,
                "is_interruption": signals.is_interruption,
                "label_end_of_turn": label,
                "source_corpus": "AMI Meeting Corpus",
                "label_source": "proxy_next_speaker_change",
            }
        )

    return rows


def write_rows(rows: list[dict[str, object]], output_path: Path) -> Path:
    if not rows:
        raise ValueError("No rows were derived from the AMI annotations.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AMI proxy end-of-turn data.")
    parser.add_argument("--ami-root", type=Path, default=DEFAULT_AMI_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-pause-ms", type=int, default=200)
    parser.add_argument("--max-pause-ms", type=int, default=3000)
    parser.add_argument("--context-words", type=int, default=18)
    parser.add_argument("--max-rows-per-label", type=int, default=3000)
    args = parser.parse_args()

    try:
        events = load_ami_words(args.ami_root)
        rows = derive_rows(
            events,
            min_pause_ms=args.min_pause_ms,
            max_pause_ms=args.max_pause_ms,
            context_words=args.context_words,
            max_rows_per_label=args.max_rows_per_label,
        )
        output_path = write_rows(rows, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"AMI preparation failed: {exc}", file=sys.stderr)
        print("See data/ami/README.md for download and folder instructions.", file=sys.stderr)
        raise SystemExit(1) from exc

    label_counts = {0: 0, 1: 0}
    for row in rows:
        label_counts[int(row["label_end_of_turn"])] += 1
    print(f"Wrote {len(rows)} AMI-derived rows to {output_path}")
    print(f"Label counts: {label_counts}")


if __name__ == "__main__":
    main()
