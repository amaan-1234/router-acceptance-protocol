"""LLM grader ensemble over ELLIPSE essays (Phase 2 grading-uncertainty track).
OpenAI-compatible: routes to local vLLM ports OR the ASU endpoint via --base-url.
For each essay, asks K persona graders for an Overall proficiency score (1.0-5.0);
the spread is the MODEL disagreement signal vs the instructor rater_gap.

Output (per model, cached): outputs/ellipse_graders_<model>_<tag>.jsonl

Usage:
    # local vLLM (model served on a port by a slurm job)
    export LLM_API_KEY=local
    python -m run.run_ellipse_graders --model qwen25-3b \
        --base-url http://localhost:8110/v1 --key-env LLM_API_KEY --out-tag sub450
    # ASU hosted endpoint
    python -m run.run_ellipse_graders --model gemma4-31b-it \
        --base-url https://openai.rc.asu.edu/v1 --key-env ASU_API_KEY --out-tag sub450
    # reasoning model: bump tokens
    python -m run.run_ellipse_graders --model qwen36-27b-fp8 \
        --base-url https://openai.rc.asu.edu/v1 --key-env ASU_API_KEY --max-tokens 4000 --out-tag sub450
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
    OUTPUTS_DIR = config.OUTPUTS_DIR
except Exception:
    OUTPUTS_DIR = Path("outputs")

from cac.data import ellipse  # type: ignore

DEFAULT_MODEL = "gemma4-31b-it"
DEFAULT_OUT_TAG = "sub450"
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_KEY_ENV = "LLM_API_KEY"

PERSONAS = [
    ("conventions-focused", "You are a CONVENTIONS-FOCUSED grader: weigh grammar, "
                            "spelling, punctuation, and sentence-level correctness "
                            "heavily when judging overall proficiency."),
    ("ideas-focused", "You are an IDEAS-FOCUSED grader: weigh the strength, clarity, "
                      "and development of the argument and ideas heavily, giving "
                      "benefit of the doubt for surface errors."),
    ("holistic", "You are a HOLISTIC grader: judge overall communicative "
                 "effectiveness for an English-language-learner audience, "
                 "balancing all aspects equally."),
    ("comparative", "You are a COMPARATIVE grader: judge this essay's overall "
                    "proficiency relative to a typical grade 8-12 English Language "
                    "Learner writing sample."),
]

PROMPT = (
    "{disposition}\n\n"
    "You are scoring an essay written by an English Language Learner (grades 8-12) "
    "for OVERALL holistic language proficiency, on a scale from 1.0 (lowest) to "
    "5.0 (highest), in increments of 0.5. Consider cohesion, syntax, vocabulary, "
    "phraseology, grammar, and conventions together as a single overall judgment.\n\n"
    "Essay:\n{text}\n\n"
    'End your reply with ONLY a JSON object: {{"score": <number 1.0-5.0>}}.'
)


def _safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")


def _out_path(model: str, tag: str = DEFAULT_OUT_TAG) -> Path:
    return OUTPUTS_DIR / f"ellipse_graders_{_safe_name(model)}_{tag}.jsonl"


def _client(base_url: str, key_env: str):
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai --break-system-packages")
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"{key_env} not set. Run: export {key_env}=...")
    return OpenAI(base_url=base_url, api_key=key, timeout=90.0, max_retries=2)


def _parse_score(text: str) -> float | None:
    cands = re.findall(r"\{[^{}]*\}", text or "", re.DOTALL)
    for m in reversed(cands):  # last valid object = final answer (reasoning-safe)
        try:
            v = float(json.loads(m)["score"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        return max(ellipse.SCORE_MIN, min(ellipse.SCORE_MAX, v))
    return None


def _grade_once(client, model, persona_text, it, temperature, max_tokens, max_retries=5):
    msg = PROMPT.format(disposition=persona_text, text=it.text)
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, max_tokens=max_tokens,
                          messages=[{"role": "user", "content": msg}])
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            ch = resp.choices[0].message
            txt = ch.content or getattr(ch, "reasoning_content", "") or ""
            return _parse_score(txt)
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{max_retries} in {wait}s] "
                  f"{type(e).__name__}: {str(e)[:100]}")
            time.sleep(wait)
    return None


def _load_done(out_path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                continue
    return done


def run(model, limit, temperature, full, base_url, key_env, max_tokens, out_tag):
    items = ellipse.load(drop_identifying=True)
    if not full:
        items = ellipse.stratified_subset(items, n=450)
    if limit:
        items = items[:limit]
    out_path = _out_path(model, out_tag)
    done = _load_done(out_path)
    client = _client(base_url, key_env)
    print(f"[ellipse] model={model} base={base_url} personas={len(PERSONAS)} "
          f"temp={temperature} items={len(items)} (done: {len(done)}) -> {out_path.name}")

    with open(out_path, "a", encoding="utf-8") as f:
        for n_done, it in enumerate(items, 1):
            if it.uid in done:
                continue
            scores = []
            for _, persona_text in PERSONAS:
                s = _grade_once(client, model, persona_text, it, temperature, max_tokens)
                if s is not None:
                    scores.append(s)
            if len(scores) < 2:
                print(f"  ! uid={it.uid}: only {len(scores)} valid scores, skipping")
                continue
            arr = np.array(scores)
            rec = {
                "uid": it.uid,
                "model": model,
                "model_scores": [round(float(x), 3) for x in scores],
                "model_mean": round(float(arr.mean()), 4),
                "model_std": round(float(arr.std()), 4),
                "model_gap": round(float(arr.max() - arr.min()), 4),
                "overall_1": it.overall_1, "overall_2": it.overall_2,
                "rater_gap": it.rater_gap, "human_grade": it.human_grade,
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            if n_done % 25 == 0:
                print(f"  [{n_done}/{len(items)}] uid={it.uid} "
                      f"model_scores={rec['model_scores']} rater_gap={it.rater_gap}")
    print(f"[ellipse] done -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--full", action="store_true",
                    help="run all 8890 essays instead of the 450-item stratified subset")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="nonzero so personas produce genuine score spread")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="raise to ~4000 for reasoning models (qwen36-27b-fp8)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--key-env", default=DEFAULT_KEY_ENV)
    ap.add_argument("--out-tag", default=DEFAULT_OUT_TAG)
    args = ap.parse_args()
    run(args.model, args.limit, args.temperature, args.full,
        args.base_url, args.key_env, args.max_tokens, args.out_tag)


if __name__ == "__main__":
    main()
