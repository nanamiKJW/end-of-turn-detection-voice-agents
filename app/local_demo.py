"""Dependency-free local browser demo for end-of-turn detection.

Use this when Streamlit is not installed or there is not enough disk space for
the Streamlit dependency stack.
"""

from __future__ import annotations

import csv
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AMI_DATA_PATH = PROJECT_ROOT / "data" / "ami_turn_samples.csv"
SYNTHETIC_DATA_PATH = PROJECT_ROOT / "data" / "sample_turns.csv"
LIGHTWEIGHT_MODEL_PATH = PROJECT_ROOT / "models" / "lightweight_eot_model.json"


QUESTION_WORDS = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
    "am",
}
BACKCHANNELS = {"yeah", "yes", "yep", "okay", "ok", "right", "sure", "mhm", "thanks"}
INCOMPLETE_LAST_TOKENS = {"to", "and", "or", "because", "with", "about", "for", "if"}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?", text.lower())


def signals(text: str) -> dict[str, object]:
    tokens = tokenize(text)
    cleaned = text.strip().lower().strip(" .?!,")
    last_token = tokens[-1] if tokens else ""
    ends_with_punctuation = int(text.strip().endswith((".", "?", "!")))
    has_question_word = int(bool(tokens) and (tokens[0] in QUESTION_WORDS or "?" in text))
    is_backchannel = int(cleaned in BACKCHANNELS)
    is_interruption = int(text.strip().endswith(("-", "--", "...")) or "[interrupted]" in text)

    score = 0.45
    if ends_with_punctuation:
        score += 0.35
    if has_question_word:
        score += 0.12
    if is_backchannel:
        score += 0.25
    if len(tokens) >= 5:
        score += 0.08
    if last_token in INCOMPLETE_LAST_TOKENS:
        score -= 0.35
    if is_interruption:
        score -= 0.30
    completeness = min(max(score, 0.0), 1.0)

    return {
        "num_words": len(tokens),
        "last_token": last_token,
        "ends_with_punctuation": ends_with_punctuation,
        "has_question_word": has_question_word,
        "syntactic_completeness_score": round(completeness, 2),
        "is_backchannel": is_backchannel,
        "is_interruption": is_interruption,
    }


def predict(text: str, pause_ms: int, threshold: float = 0.5) -> dict[str, object]:
    sig = signals(text)
    probability = 0.15
    if pause_ms >= 1000:
        probability += 0.35
    elif pause_ms >= 700:
        probability += 0.25
    elif pause_ms >= 500:
        probability += 0.12
    probability += 0.25 * float(sig["syntactic_completeness_score"])
    if sig["ends_with_punctuation"]:
        probability += 0.14
    if sig["has_question_word"] and sig["ends_with_punctuation"]:
        probability += 0.08
    if sig["is_backchannel"]:
        probability += 0.18
    if sig["is_interruption"]:
        probability -= 0.30
    if sig["last_token"] in INCOMPLETE_LAST_TOKENS:
        probability -= 0.20
    probability = min(max(probability, 0.02), 0.98)

    is_end = probability >= threshold
    reasons = []
    if pause_ms >= 700:
        reasons.append("the pause is fairly long")
    else:
        reasons.append("the pause is still short")
    if sig["last_token"] in INCOMPLETE_LAST_TOKENS:
        reasons.append(f"the last word '{sig['last_token']}' often leaves a phrase unfinished")
    if sig["ends_with_punctuation"]:
        reasons.append("the text looks sentence-final")
    if sig["is_backchannel"]:
        reasons.append("this looks like a short complete response")
    if sig["syntactic_completeness_score"] < 0.45:
        reasons.append("the completeness score is low")
    elif sig["syntactic_completeness_score"] >= 0.75:
        reasons.append("the completeness score is high")

    if is_end:
        plain_explanation = (
            "The system thinks the user has probably finished, so a voice agent could respond now."
        )
    else:
        plain_explanation = (
            "The system thinks the user is probably still speaking, so a voice agent should wait."
        )

    return {
        "label": "End of Turn" if is_end else "Likely Continuing",
        "probability": probability,
        "baseline_700ms": "End of Turn" if pause_ms >= 700 else "Likely Continuing",
        "signals": sig,
        "model_type": "rule_based_local_demo",
        "threshold": threshold,
        "plain_explanation": plain_explanation,
        "reasons": reasons,
    }


def load_lightweight_model() -> dict[str, object] | None:
    if not LIGHTWEIGHT_MODEL_PATH.exists():
        return None
    return json.loads(LIGHTWEIGHT_MODEL_PATH.read_text(encoding="utf-8"))


def gaussian_log_probability(value: float, mean: float, std: float) -> float:
    variance = std * std
    return -0.5 * math.log(2 * math.pi * variance) - ((value - mean) ** 2 / (2 * variance))


def predict_with_lightweight_model(
    text: str,
    pause_ms: int,
    threshold: float = 0.5,
) -> dict[str, object]:
    model = load_lightweight_model()
    if model is None:
        return predict(text, pause_ms, threshold)

    sig = signals(text)
    numeric_row = {
        "pause_duration_ms": float(pause_ms),
        "speech_duration_ms": float(max(250, sig["num_words"] * 360)),
        "num_words": float(sig["num_words"]),
        "ends_with_punctuation": float(sig["ends_with_punctuation"]),
        "has_question_word": float(sig["has_question_word"]),
        "syntactic_completeness_score": float(sig["syntactic_completeness_score"]),
        "is_backchannel": float(sig["is_backchannel"]),
        "is_interruption": float(sig["is_interruption"]),
    }

    tokens = tokenize(text)
    vocabulary_size = int(model["vocabulary_size"])
    scores = {}
    for label in ("0", "1"):
        score = math.log(float(model["class_priors"][label]))
        token_counts = model["token_counts"][label]
        token_total = int(model["token_totals"][label])

        for token in tokens:
            count = int(token_counts.get(token, 0))
            score += math.log((count + 1) / (token_total + vocabulary_size))

        for feature in model["numeric_features"]:
            stats = model["numeric_stats"][label][feature]
            score += gaussian_log_probability(
                numeric_row[feature],
                float(stats["mean"]),
                float(stats["std"]),
            )
        scores[label] = score

    max_score = max(scores.values())
    prob_1 = math.exp(scores["1"] - max_score)
    prob_0 = math.exp(scores["0"] - max_score)
    probability = prob_1 / (prob_0 + prob_1)
    is_end = probability >= threshold

    if is_end:
        plain_explanation = "The model would let the assistant respond at this point."
    else:
        plain_explanation = "The model would keep waiting because the turn looks unfinished."

    reasons = [
        f"trained on {model['training_rows']} AMI-derived examples",
        f"pause after speech: {pause_ms} ms",
        f"last token: '{sig['last_token'] or 'none'}'",
        f"completeness score: {sig['syntactic_completeness_score']}",
    ]

    return {
        "label": "End of Turn" if is_end else "Likely Continuing",
        "probability": probability,
        "baseline_700ms": "End of Turn" if pause_ms >= 700 else "Likely Continuing",
        "signals": sig,
        "model_type": "lightweight_trained_model",
        "threshold": threshold,
        "plain_explanation": plain_explanation,
        "reasons": reasons,
    }


def dataset_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False}
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = [int(row["label_end_of_turn"]) for row in rows]
    return {
        "exists": True,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "rows": len(rows),
        "end_turn_rate": round(sum(labels) / len(labels), 3) if labels else 0.0,
    }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>End-of-Turn Detection Demo</title>
  <style>
    :root { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f7f8fa; }
    body { margin: 0; }
    main { max-width: 1080px; margin: 0 auto; padding: 34px 22px 56px; }
    h1 { margin: 0 0 8px; font-size: 32px; letter-spacing: 0; }
    p { line-height: 1.55; color: #4b5563; }
    .grid { display: grid; grid-template-columns: 1.15fr .85fr; gap: 18px; margin-top: 22px; }
    .panel { background: #fff; border: 1px solid #dfe4ea; border-radius: 8px; padding: 18px; }
    label { display: block; font-weight: 650; margin: 14px 0 8px; }
    textarea { width: 100%; min-height: 118px; box-sizing: border-box; border: 1px solid #cfd7df; border-radius: 6px; padding: 12px; font: inherit; }
    select { width: 100%; box-sizing: border-box; border: 1px solid #cfd7df; border-radius: 6px; padding: 10px; font: inherit; background: white; }
    input[type=range] { width: 100%; }
    button { background: #245b7a; color: white; border: 0; border-radius: 6px; padding: 10px 14px; font-weight: 650; cursor: pointer; margin-top: 14px; }
    button.secondary { background: #eef2f5; color: #24313d; margin: 4px 6px 4px 0; }
    .metric { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 16px; }
    .metric div { background: #f4f7fa; border-radius: 6px; padding: 12px; }
    .metric strong { display: block; font-size: 20px; margin-top: 4px; color: #111827; }
    .result { font-size: 26px; font-weight: 800; margin: 8px 0; }
    .badge { display: inline-block; padding: 5px 9px; border-radius: 999px; background: #e8f0f5; color: #245b7a; font-weight: 650; font-size: 13px; }
    .plain { background: #fbf7ed; border: 1px solid #ead8ae; border-radius: 6px; padding: 12px; color: #3d3521; }
    .why { margin: 10px 0 0; padding-left: 20px; color: #354150; }
    .why li { margin: 7px 0; }
    .small { font-size: 14px; color: #5b6876; }
    pre { white-space: pre-wrap; background: #f4f7fa; padding: 12px; border-radius: 6px; overflow: auto; }
    @media (max-width: 850px) { .grid, .metric { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>End-of-Turn Detection Demo</h1>
  <p>This is a small prototype for deciding whether a voice assistant should answer now or keep listening. It uses a lightweight model trained on AMI-derived meeting examples, plus a few transparent transcript and timing features.</p>
  <section class="metric">
    <div>AMI-derived examples<strong id="amiRows">...</strong></div>
    <div>End-turn labels<strong id="amiRate">...</strong></div>
    <div>Model<strong id="modelMode">...</strong></div>
  </section>
  <section class="grid">
    <div class="panel">
      <h2>Try a partial turn</h2>
      <button class="secondary" onclick="setExample('I want to book a flight to', 400)">Incomplete request</button>
      <button class="secondary" onclick="setExample('I want to book a flight to Paris tomorrow.', 900)">Complete request</button>
      <button class="secondary" onclick="setExample('Okay, thanks.', 350)">Short answer</button>
      <button class="secondary" onclick="setExample('The reason I called is because', 850)">Long pause mid-turn</button>
      <label for="text">Partial transcript</label>
      <textarea id="text">I want to book a flight to</textarea>
      <label for="mode">Response style</label>
      <select id="mode" onchange="runPredict()">
        <option value="0.50">Balanced</option>
        <option value="0.35">Respond earlier</option>
        <option value="0.65">Wait longer</option>
      </select>
      <label for="pause">Pause duration: <span id="pauseValue">400</span> ms</label>
      <input id="pause" type="range" min="100" max="1800" step="50" value="400" oninput="pauseValue.textContent=this.value">
      <button onclick="runPredict()">Update result</button>
    </div>
    <div class="panel">
      <h2>Prediction</h2>
      <span class="badge">Assistant action</span>
      <div class="result" id="label">...</div>
      <p class="plain" id="plainExplanation"></p>
      <p id="probability"></p>
      <p id="baseline"></p>
      <h3>What influenced it?</h3>
      <ul class="why" id="reasons"></ul>
      <h3>Feature values</h3>
      <p class="small">Shown for transparency. These are not meant to be perfect linguistic analyses.</p>
      <pre id="signals"></pre>
    </div>
  </section>
</main>
<script>
async function loadSummary() {
  const response = await fetch('/summary');
  const data = await response.json();
  document.getElementById('amiRows').textContent = data.ami.exists ? data.ami.rows : 'missing';
  document.getElementById('amiRate').textContent = data.ami.exists ? Math.round(data.ami.end_turn_rate * 100) + '%' : 'missing';
  document.getElementById('modelMode').textContent = data.model.exists ? 'trained lightweight model' : 'fallback';
}
function setExample(text, pause) {
  document.getElementById('text').value = text;
  document.getElementById('pause').value = pause;
  document.getElementById('pauseValue').textContent = pause;
  runPredict();
}
async function runPredict() {
  const text = document.getElementById('text').value;
  const pause = document.getElementById('pause').value;
  const threshold = document.getElementById('mode').value;
  const response = await fetch('/predict?text=' + encodeURIComponent(text) + '&pause_ms=' + encodeURIComponent(pause) + '&threshold=' + encodeURIComponent(threshold));
  const data = await response.json();
  document.getElementById('label').textContent = data.label;
  document.getElementById('plainExplanation').textContent = data.plain_explanation;
  document.getElementById('probability').textContent = 'Model score: ' + Math.round(data.probability * 100) + '% end-of-turn';
  document.getElementById('probability').textContent += ' | decision threshold: ' + Math.round(data.threshold * 100) + '%';
  document.getElementById('baseline').textContent = 'Pause-only baseline: ' + data.baseline_700ms + ' at a 700 ms threshold.';
  document.getElementById('reasons').innerHTML = data.reasons.map(reason => '<li>' + reason + '</li>').join('');
  document.getElementById('signals').textContent = JSON.stringify(data.signals, null, 2);
}
loadSummary();
runPredict();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.respond(HTML, "text/html")
        elif parsed.path == "/summary":
            self.respond_json(
                {
                    "ami": dataset_summary(AMI_DATA_PATH),
                    "synthetic": dataset_summary(SYNTHETIC_DATA_PATH),
                    "model": {"exists": LIGHTWEIGHT_MODEL_PATH.exists()},
                }
            )
        elif parsed.path == "/predict":
            query = parse_qs(parsed.query)
            text = query.get("text", [""])[0]
            pause_ms = int(query.get("pause_ms", ["700"])[0])
            threshold = float(query.get("threshold", ["0.5"])[0])
            self.respond_json(predict_with_lightweight_model(text, pause_ms, threshold=threshold))
        else:
            self.send_response(404)
            self.end_headers()

    def respond(self, body: str, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def respond_json(self, data: dict[str, object]) -> None:
        self.respond(json.dumps(data), "application/json")


def main() -> None:
    server = ThreadingHTTPServer(("localhost", 8501), Handler)
    print("Local demo running at http://localhost:8501")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
