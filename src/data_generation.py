"""Generate a synthetic portfolio dataset for end-of-turn detection."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from preprocessing import extract_text_signals


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sample_turns.csv"


EXAMPLES = [
    ("I want to book a flight to", 430, 1710, 0),
    ("I want to book a flight to Paris.", 920, 2300, 1),
    ("Can you help me with my visa application?", 850, 2240, 1),
    ("Because I was thinking that maybe", 350, 1550, 0),
    ("Yeah.", 620, 340, 1),
    ("Wait, I mean", 280, 760, 0),
    ("I need to change my appointment from Tuesday to", 460, 2680, 0),
    ("I need to change my appointment from Tuesday to Friday.", 980, 3260, 1),
    ("What time does the last train leave?", 760, 1820, 1),
    ("What time does the last train", 410, 1360, 0),
    ("Okay.", 700, 260, 1),
    ("Right", 520, 210, 1),
    ("Well I was hoping you could", 330, 1490, 0),
    ("Could you send the confirmation by email?", 940, 1980, 1),
    ("Could you send the confirmation by", 390, 1510, 0),
    ("I am looking for a quiet hotel near the station.", 1120, 3300, 1),
    ("I am looking for a quiet hotel near", 460, 2170, 0),
    ("No, that's not what I meant.", 870, 1490, 1),
    ("No, that's not", 260, 760, 0),
    ("Um I think the reference number is", 310, 1900, 0),
    ("The reference number is A five three nine.", 900, 2460, 1),
    ("Actually could you check another date?", 810, 1740, 1),
    ("Actually could you check another", 420, 1290, 0),
    ("I live in Barcelona but I am moving to", 450, 2380, 0),
    ("I live in Barcelona but I am moving to Lyon next month.", 1020, 3600, 1),
    ("Mhm.", 580, 200, 1),
    ("Sure.", 690, 240, 1),
    ("Thanks.", 760, 220, 1),
    ("I was wondering if", 290, 920, 0),
    ("I was wondering if you have appointments tomorrow.", 880, 2380, 1),
    ("Can I pay with a student discount?", 780, 1700, 1),
    ("Can I pay with a student", 360, 1280, 0),
    ("The thing is", 300, 800, 0),
    ("The thing is, I already uploaded the document.", 920, 2400, 1),
    ("Sorry, can you repeat that?", 740, 1340, 1),
    ("Sorry, can you", 270, 620, 0),
    ("I need it before Monday because", 390, 1530, 0),
    ("I need it before Monday because my flight is early.", 980, 2920, 1),
    ("So the address is", 340, 920, 0),
    ("So the address is 14 Oxford Road.", 820, 1780, 1),
    ("Do I need to bring my passport?", 830, 1650, 1),
    ("Do I need to bring my", 380, 1180, 0),
    ("Yes", 530, 180, 1),
    ("No.", 610, 180, 1),
    ("I mean the second option", 510, 1130, 1),
    ("I mean the second", 300, 810, 0),
    ("And then after that", 330, 960, 0),
    ("And then after that I want to cancel the old booking.", 960, 3140, 1),
    ("My account number is seven four two", 520, 2210, 0),
    ("My account number is seven four two eight.", 910, 2460, 1),
    ("Wait, I mean the morning slot.", 740, 1420, 1),
    ("Wait, I mean the morning", 340, 1100, 0),
    ("Is there a direct bus to the airport?", 870, 1890, 1),
    ("Is there a direct bus to", 410, 1280, 0),
    ("I need help with my residence permit renewal.", 1030, 2510, 1),
    ("I need help with my residence permit", 470, 1960, 0),
    ("The cheaper ticket is fine.", 870, 1220, 1),
    ("The cheaper ticket is", 310, 780, 0),
    ("Could we maybe move it to next", 360, 1420, 0),
    ("Could we maybe move it to next Wednesday?", 900, 2120, 1),
    ("I am not sure if that works", 620, 1770, 1),
    ("I am not sure if", 280, 910, 0),
    ("There are three people travelling with me.", 920, 2110, 1),
    ("There are three people travelling with", 400, 1640, 0),
    ("Okay that makes sense.", 720, 860, 1),
    ("Okay so", 250, 420, 0),
    ("You know I was trying to", 300, 1210, 0),
    ("You know I was trying to update the form online.", 900, 2780, 1),
    ("Can you check whether the office is open on Saturdays?", 920, 2860, 1),
    ("Can you check whether the office is open on", 430, 2330, 0),
    ("I booked it under my university email.", 890, 2000, 1),
    ("I booked it under my", 340, 990, 0),
    ("Please cancel the appointment.", 840, 1050, 1),
    ("Please cancel the", 310, 620, 0),
    ("The problem started after I changed my password.", 950, 2460, 1),
    ("The problem started after", 340, 910, 0),
    ("Could you tell me where to upload the file?", 900, 2340, 1),
    ("Could you tell me where to upload", 430, 1830, 0),
    ("I have already tried restarting the app.", 890, 1860, 1),
    ("I have already tried", 320, 760, 0),
    ("It should arrive before the end of the week.", 1010, 2600, 1),
    ("It should arrive before the end of", 460, 2010, 0),
    ("Sorry I lost the connection.", 830, 1310, 1),
    ("Sorry I lost the", 330, 740, 0),
    ("I need a receipt for my reimbursement.", 860, 1830, 1),
    ("I need a receipt for", 360, 980, 0),
    ("Is that included in the price?", 780, 1460, 1),
    ("Is that included in", 330, 910, 0),
    ("No but I can send it later.", 870, 1710, 1),
    ("No but I can", 290, 740, 0),
    ("The appointment was cancelled without notice.", 900, 2020, 1),
    ("The appointment was cancelled without", 410, 1500, 0),
    ("I think so.", 690, 610, 1),
    ("I think", 260, 360, 0),
    ("That's all.", 760, 520, 1),
    ("That's", 230, 240, 0),
    ("I need to speak with someone from admissions.", 920, 2240, 1),
    ("I need to speak with someone from", 430, 1700, 0),
    ("Can you hold on for a second?", 760, 1400, 1),
    ("Can you hold on for", 330, 920, 0),
    ("I was told to call this number.", 840, 1700, 1),
    ("I was told to call", 320, 960, 0),
    ("My train leaves at half past six.", 880, 1870, 1),
    ("My train leaves at", 310, 830, 0),
    ("The invoice should be in my name.", 840, 1780, 1),
    ("The invoice should be in", 350, 1110, 0),
    ("I uploaded the document yesterday, but", 390, 1750, 0),
    ("I uploaded the document yesterday, but it still says pending.", 950, 3120, 1),
    ("Could you maybe- [interrupted]", 180, 620, 0),
    ("I need the blue- [interrupted]", 160, 710, 0),
    ("The date is the twenty- [interrupted]", 190, 980, 0),
    ("Yes, exactly.", 690, 520, 1),
    ("Right, that is correct.", 780, 900, 1),
    ("I want to ask about", 820, 970, 0),
    ("The reason I called is because", 910, 1510, 0),
    ("Could you help me with", 760, 1180, 0),
    ("I am trying to find", 880, 1120, 0),
    ("If possible I would like to", 790, 1400, 0),
    ("My question is about", 840, 900, 0),
    ("The form asks for", 780, 1060, 0),
    ("I need to travel on", 860, 1260, 0),
    ("Yes.", 360, 190, 1),
    ("Okay, thanks.", 420, 410, 1),
    ("No thanks.", 390, 360, 1),
    ("That's correct.", 430, 470, 1),
    ("Perfect.", 380, 240, 1),
    ("Not today.", 450, 410, 1),
    ("Tomorrow morning.", 470, 520, 1),
    ("Two people.", 410, 430, 1),
    ("In Madrid.", 460, 440, 1),
    ("By email.", 400, 380, 1),
    ("That works.", 430, 430, 1),
    ("I agree.", 420, 360, 1),
]


def jitter(value: int, label: int) -> int:
    """Add small deterministic-looking variation without changing the task."""

    offset = random.randint(-80, 120) if label else random.randint(-90, 70)
    return max(80, value + offset)


def build_rows(seed: int = 7) -> list[dict[str, object]]:
    random.seed(seed)
    rows = []
    for index, (text, pause_ms, speech_ms, label) in enumerate(EXAMPLES, start=1):
        signals = extract_text_signals(text)
        rows.append(
            {
                "turn_id": f"turn_{index:03d}",
                "utterance_text": text,
                "pause_duration_ms": jitter(pause_ms, label),
                "speech_duration_ms": jitter(speech_ms, label),
                "num_words": signals.num_words,
                "last_token": signals.last_token,
                "ends_with_punctuation": signals.ends_with_punctuation,
                "has_question_word": signals.has_question_word,
                "syntactic_completeness_score": signals.syntactic_completeness_score,
                "is_backchannel": signals.is_backchannel,
                "is_interruption": signals.is_interruption,
                "label_end_of_turn": label,
            }
        )
    return rows


def write_dataset(path: Path = DATA_PATH) -> Path:
    rows = build_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    output_path = write_dataset()
    print(f"Wrote synthetic dataset to {output_path}")
