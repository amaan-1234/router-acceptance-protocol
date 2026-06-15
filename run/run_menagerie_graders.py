"""LLM grader ensemble over Menagerie (Phase 2 code-grading uncertainty track).

Mirrors run_mohler_graders.py / run_frontier.py: cached, resumable, safe key handling.
For each (submission, criterion), K grader personas read the submission's Java source
and return a 0-13 ordinal score on that one criterion. The spread of the K model scores
is the MODEL disagreement signal, correlated against the 4-grader human std.

One cache row per (submission, criterion). 279 submissions x 4 criteria x K personas
~= 4,460 calls at K=4. Cached per row, so a reaped run resumes for free.

Criteria match the Menagerie rubric exactly: Correctness, Code Elegance, Readability,
Documentation. The per-criterion rubric text is the dataset's own definition so the
model grades on the same construct the human assessors did.

Output: outputs/menagerie_graders.jsonl
  {uid, submission_id, criterion, model_scores:[K], model_mean, model_std, model_range,
   human_scores:[4], human_std, human_mean}

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...        # via read -s, never inline
    python -m run.run_menagerie_graders --limit 8      # smoke (2 submissions x 4 criteria)
    python -m run.run_menagerie_graders                # full 1116 items
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

try:
    from cac import config
    OUT_PATH = config.OUTPUTS_DIR / "menagerie_graders.jsonl"
except Exception:
    OUT_PATH = Path("outputs/menagerie_graders.jsonl")

from cac.data import menagerie  # type: ignore

DEFAULT_MODEL = "claude-sonnet-4-6"

# Menagerie's own criterion definitions (from the dataset / paper).
RUBRIC = {
    "Correctness": "how well the submission meets the assignment requirements "
                   "(a predator/prey simulator: multiple species, shared food source, "
                   "time-of-day tracking).",
    "Code Elegance": "maintainability: correct use of functions and classes, "
                     "appropriate object-oriented design, no needless duplication.",
    "Readability": "how readable the source is: meaningful identifier names, "
                   "consistent whitespace and indentation, clear structure.",
    "Documentation": "whether the documentation is well written and organized and "
                     "clearly explains what the code accomplishes.",
}

# Grader personas (heterogeneous panel), same idea as the Mohler runner.
PERSONAS = [
    ("strict", "You are a STRICT grader: award the top grade only when the criterion "
               "is met fully and without exception."),
    ("lenient", "You are a LENIENT grader: reward clear effort and partial success, "
                "giving benefit of the doubt."),
    ("experienced", "You are an EXPERIENCED senior grader: weigh the criterion against "
                    "professional software-engineering norms."),
    ("rubric_literal", "You are a LITERAL grader: judge strictly against the stated "
                       "criterion definition, nothing more."),
]

# 0-13 ordinal letter scale, matching menagerie.GRADE_MAP (F=0 .. A++=13).
SCALE_LEGEND = ("0=F, 1=D-, 2=D, 3=D+, 4=C-, 5=C, 6=C+, 7=B-, 8=B, 9=B+, "
                "10=A-, 11=A, 12=A+, 13=A++")

PROMPT = (
    "{disposition}\n\n"
    "You are grading a second-semester CS1 Java assignment (a predator/prey simulator) "
    "on ONE criterion only.\n\n"
    "Criterion -- {criterion}: {crit_def}\n\n"
    "Grade on this 0 to 13 ordinal scale ({scale}).\n\n"
    "Submission source files:\n{code}\n\n"
    'Output ONLY a JSON object: {{"score": <integer 0-13>}}. Nothing else.'
)


def _client():
    try:
        import anthropic
    except ImportError:
        raise SystemExit("pip install anthropic --break-system-packages")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-ant-...")
    return anthropic.Anthropic(api_key=key)


def _parse_score(text: str):
    m = re.search(r"\{[^}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        v = float(json.loads(m.group(0))["score"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return max(0.0, min(menagerie.GRADE_MAX, v))


def _format_code(java: dict) -> str:
    return "\n\n".join(f"// ===== {name} =====\n{src}" for name, src in java.items())


def _grade_once(client, model, persona_text, criterion, code, temperature, max_retries=5):
    msg = PROMPT.format(disposition=persona_text, criterion=criterion,
                        crit_def=RUBRIC[criterion], scale=SCALE_LEGEND, code=code)
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, max_tokens=20,
                          messages=[{"role": "user", "content": msg}])
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.messages.create(**kwargs)
            txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return _parse_score(txt)
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{max_retries} in {wait}s] "
                  f"{type(e).__name__}: {str(e)[:100]}")
            time.sleep(wait)
    return None


def _load_done() -> set:
    if not OUT_PATH.exists():
        return set()
    done = set()
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                continue
    return done


def run(model, limit, temperature):
    items = menagerie.load_items()
    if limit:
        items = items[:limit]
    done = _load_done()
    client = _client()
    print(f"[menagerie] model={model} personas={len(PERSONAS)} temp={temperature} "
          f"items={len(items)} (already done: {len(done)})")

    # cache Java per submission so the 4 criteria of one submission share one read
    java_cache: dict[str, str] = {}

    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for n_done, it in enumerate(items, 1):
            uid = f"{it.submission_id}::{it.criterion}"
            if uid in done:
                continue
            if it.submission_id not in java_cache:
                src = menagerie.load_java_sources(it.submission_id)
                java_cache[it.submission_id] = _format_code(src) if src else ""
            code = java_cache[it.submission_id]
            if not code:
                print(f"  ! {uid}: no Java source, skipping")
                continue
            scores = []
            for _, persona_text in PERSONAS:
                s = _grade_once(client, model, persona_text, it.criterion, code, temperature)
                if s is not None:
                    scores.append(s)
            if len(scores) < 2:
                print(f"  ! {uid}: only {len(scores)} valid scores, skipping")
                continue
            arr = np.array(scores)
            rec = {
                "uid": uid, "submission_id": it.submission_id, "criterion": it.criterion,
                "model_scores": [round(float(x), 2) for x in scores],
                "model_mean": round(float(arr.mean()), 4),
                "model_std": round(float(arr.std()), 4),
                "model_range": round(float(arr.max() - arr.min()), 4),
                "human_scores": it.grader_scores,
                "human_std": round(it.rater_std, 4),
                "human_mean": round(it.adjudicated, 4),
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            if n_done % 20 == 0:
                print(f"  [{n_done}/{len(items)}] {uid} "
                      f"model={rec['model_scores']} human_std={rec['human_std']}")
    print(f"[menagerie] done -> {OUT_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()
    temp = None if "opus" in args.model.lower() else args.temperature
    run(args.model, args.limit, temp)


if __name__ == "__main__":
    main()
