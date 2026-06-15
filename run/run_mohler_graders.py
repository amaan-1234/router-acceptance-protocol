"""LLM grader ensemble over Mohler ASAG (Phase 2 grading-uncertainty track).

Mirrors run_frontier.py: cached, resumable, safe key handling. For each student
answer, asks K independent grader "personas" (varied grading dispositions, like a
heterogeneous human grading panel) for a 0-5 score in a single structured call each.
The spread of those K model scores is the MODEL disagreement signal we correlate
against the instructor rater gap.

Why personas rather than K different model checkpoints: the Phase 1 ensemble used
different small models to get heterogeneous disagreement. Here we elicit the same
effect cheaply by varying the grader's stated disposition (strict / lenient /
holistic / literal) at temperature, which produces genuine score spread on ambiguous
answers while staying near-deterministic on clear ones — exactly the property the
disagreement signal needs.

Output (cached, one line per item):  outputs/mohler_graders.jsonl
  {uid, model_scores:[K floats], model_mean, model_std, model_gap,
   score_me, score_other, rater_gap, score_avg}

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...        # via read -s, never inline
    python -m run.run_mohler_graders --limit 10     # smoke test
    python -m run.run_mohler_graders                # full 2273
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
    OUT_PATH = config.OUTPUTS_DIR / "mohler_graders.jsonl"
except Exception:
    OUT_PATH = Path("outputs/mohler_graders.jsonl")

from cac.data import mohler  # type: ignore

DEFAULT_MODEL = "claude-sonnet-4-6"

# K grader personas — a heterogeneous panel. Disposition text is the only thing
# that varies; all see the same question / reference / answer.
PERSONAS = [
    ("strict", "You are a STRICT grader: award full marks only when the answer is "
               "complete and precise; deduct for any missing or vague element."),
    ("lenient", "You are a LENIENT grader: reward partial understanding and give "
                "benefit of the doubt where the core idea is present."),
    ("holistic", "You are a HOLISTIC grader: judge overall understanding rather than "
                 "specific keywords."),
    ("literal", "You are a LITERAL grader: compare the answer closely against the "
                "reference answer's specific content."),
]

PROMPT = (
    "{disposition}\n\n"
    "Grade the student answer on a 0 to 5 scale (0=completely wrong, 5=fully correct). "
    "Decimals allowed.\n\n"
    "Question: {question}\n"
    "Reference answer: {desired}\n"
    "Student answer: {student}\n\n"
    'Output ONLY a JSON object: {{"score": <number 0-5>}}. Nothing else.'
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


def _parse_score(text: str) -> float | None:
    m = re.search(r"\{[^}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        v = float(json.loads(m.group(0))["score"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return max(0.0, min(mohler.SCORE_MAX, v))  # clamp into range


def _grade_once(client, model, persona_text, it, temperature, max_retries=5):
    msg = PROMPT.format(disposition=persona_text, question=it.question,
                        desired=it.desired_answer, student=it.student_answer)
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
    items = mohler.load()
    if limit:
        items = items[:limit]
    done = _load_done()
    client = _client()
    print(f"[mohler] model={model} personas={len(PERSONAS)} temp={temperature} "
          f"items={len(items)} (already done: {len(done)})")

    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for n_done, it in enumerate(items, 1):
            if it.uid in done:
                continue
            scores = []
            for _, persona_text in PERSONAS:
                s = _grade_once(client, model, persona_text, it, temperature)
                if s is not None:
                    scores.append(s)
            if len(scores) < 2:
                # not enough graders responded to measure disagreement; skip (don't cache)
                print(f"  ! uid={it.uid}: only {len(scores)} valid scores, skipping")
                continue
            arr = np.array(scores)
            rec = {
                "uid": it.uid,
                "model_scores": [round(float(x), 3) for x in scores],
                "model_mean": round(float(arr.mean()), 4),
                "model_std": round(float(arr.std()), 4),
                "model_gap": round(float(arr.max() - arr.min()), 4),
                "score_me": it.score_me, "score_other": it.score_other,
                "rater_gap": it.rater_gap, "score_avg": it.score_avg,
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            if n_done % 25 == 0:
                print(f"  [{n_done}/{len(items)}] uid={it.uid} "
                      f"model_scores={rec['model_scores']} rater_gap={it.rater_gap}")
    print(f"[mohler] done -> {OUT_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="nonzero so personas produce genuine score spread on ambiguous answers")
    args = ap.parse_args()
    temp = None if "opus" in args.model.lower() else args.temperature
    run(args.model, args.limit, temp)


if __name__ == "__main__":
    main()
