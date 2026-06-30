"""Capacity-sweep runner: stated-distribution uncertainty per model on the ChaosNLI
frozen 450 subset, via any OpenAI-compatible endpoint (a hosted OpenAI-compatible endpoint).

Parallel to run_frontier.py but model-agnostic. For each ChaosNLI item it asks ONE
model for its class-probability distribution over the 3 NLI labels, in a single call.
Run it once per model in the capacity sweep (0.8B -> 27B -> 122B via a hosted endpoint; 2B/4B/9B via
the local HF runner), then eval_sweep.py traces corr(model_entropy, H_human) against
model size and computes AUROC / risk-coverage / pairwise error-correlation.

Why the SAME frozen 450 subset as run_frontier: every sweep point and the Sonnet
frontier control (r=0.515) then sit on identical items -> the r-vs-size curve is a fair,
within-design comparison, not confounded by different item samples.

Two endpoint-specific fixes vs the Anthropic frontier runner:
  1. Reads response from `content` OR `reasoning_content` -- some thinking models
     (e.g. qwen35-0p8b) emit the answer in the reasoning channel with content=null.
  2. Brace-scanning parser picks the LAST valid {e,n,c} JSON object, so a reasoning
     trace full of stray braces still yields the final stated distribution.

Usage:
    export LLM_API_KEY=...                                  # API key
    python run_ensemble_api.py --model qwen35-0p8b --limit 5     # smoke test
    python run_ensemble_api.py --model qwen35-0p8b              # full 450
    python run_ensemble_api.py --model qwen35-0p8b --analyze-only

Caching: per-item results append to outputs/sweep_chaosnli_<model>.jsonl and flush
immediately; on restart, items already present are skipped -> no re-querying.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from cac import config
from cac.data.labels_nli import NLI_CLASSES

SUBSET_PATH = config.OUTPUTS_DIR / "chaosnli_frontier_subset_450.json"  # 450 stratified, matches Sonnet control
DEFAULT_OUT_TAG = "sub450"
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_KEY_ENV = "LLM_API_KEY"

PROMPT = (
    "You are an expert annotator for natural language inference. Given a premise and "
    "hypothesis, humans often disagree, so express your judgment as a probability "
    "distribution over the three labels rather than a single answer.\n\n"
    "Premise: {premise}\nHypothesis: {hypothesis}\n\n"
    "Output ONLY a JSON object with three keys that sum to 1.0, e.g. "
    '{{"entailment": 0.6, "neutral": 0.3, "contradiction": 0.1}}. '
    "Reflect genuine uncertainty: if the relationship is ambiguous, spread the mass; "
    "if it is clear, concentrate it. End your reply with the JSON object."
)


def _safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")


def _out_path(model: str, tag: str = DEFAULT_OUT_TAG):
    return config.OUTPUTS_DIR / f"sweep_chaosnli_{_safe_name(model)}_{tag}.jsonl"


def _client(base_url: str, key_env: str):
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai --break-system-packages")
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"{key_env} not set. Run: export {key_env}=... (then re-export so "
                         f"Python sees it: export {key_env}=\"${key_env}\")")
    return OpenAI(base_url=base_url, api_key=key, timeout=90.0, max_retries=2)


def _parse_dist(text: str) -> np.ndarray | None:
    """Scan all {...} objects, return the LAST one that parses to a valid {e,n,c} dist."""
    if not text:
        return None
    for cand in reversed(re.findall(r"\{[^{}]*\}", text, re.DOTALL)):
        try:
            obj = json.loads(cand)
            v = np.array([float(obj[c]) for c in NLI_CLASSES], dtype=np.float64)
        except Exception:
            continue
        s = v.sum()
        if np.isfinite(s) and s > 0:
            return v / s
    return None


def _call(client, model, premise, hypothesis, temperature, max_tokens, max_retries=5):
    msg = PROMPT.format(premise=premise, hypothesis=hypothesis)
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, max_tokens=max_tokens,
                          messages=[{"role": "user", "content": msg}])
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            m = resp.choices[0].message
            text = (m.content or getattr(m, "reasoning_content", None) or "")
            return _parse_dist(text), text
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{max_retries} in {wait}s] {type(e).__name__}: {str(e)[:120]}")
            time.sleep(wait)
    return None, ""


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


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


def run(model, limit, temperature, max_tokens, base_url, key_env, subset_path, out_tag):
    items = json.load(open(subset_path, encoding="utf-8"))
    if limit:
        items = items[:limit]
    out_path = _out_path(model, out_tag)
    done = _load_done(out_path)
    client = _client(base_url, key_env)
    print(f"[sweep] model={model} subset={Path(subset_path).name} base={base_url} "
          f"temp={temperature} max_tokens={max_tokens} "
          f"items={len(items)} (already done: {len(done)}) -> {out_path.name}")

    with open(out_path, "a", encoding="utf-8") as f:
        for n_done, it in enumerate(items, 1):
            if it["uid"] in done:
                continue
            dist, raw = _call(client, model, it["premise"], it["hypothesis"], temperature, max_tokens)
            if dist is None:
                dist = np.full(len(NLI_CLASSES), 1.0 / len(NLI_CLASSES))
                parse_ok = False
            else:
                parse_ok = True
            rec = {
                "uid": it["uid"], "idx": it["idx"], "model": model,
                "model_dist": [round(float(x), 4) for x in dist],
                "model_argmax": int(dist.argmax()),
                "model_entropy_bits": round(_entropy_bits(dist), 4),
                "parse_ok": parse_ok,
                "raw": raw,
                "human_dist": it["human_dist"],
                "human_entropy": it["entropy"],
                "human_argmax": int(np.argmax(it["human_dist"])),
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            if n_done % 20 == 0:
                print(f"  [{n_done}/{len(items)}] uid={it['uid']} dist={rec['model_dist']} ok={parse_ok}")
    analyze(model, out_tag)


def analyze(model, out_tag=DEFAULT_OUT_TAG):
    out_path = _out_path(model, out_tag)
    recs = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    if not recs:
        print("[sweep] no results yet."); return
    bad = sum(1 for r in recs if not r.get("parse_ok", True))
    ma = np.array([r["model_argmax"] for r in recs])
    ha = np.array([r["human_argmax"] for r in recs])
    H = np.array([r["human_entropy"] for r in recs])
    mH = np.array([r["model_entropy_bits"] for r in recs])

    print("\n" + "=" * 60)
    print(f"SWEEP POINT  model={model}  (N={len(recs)}, parse failures={bad})")
    print("=" * 60)
    print(f"  accuracy vs human-majority        : {(ma==ha).mean():.3f}")
    if mH.std() > 1e-9:
        r = float(np.corrcoef(mH, H)[0, 1])
        print(f"  corr(model entropy, H_human)      : {r:.3f}")
        print(f"    (frontier Sonnet control on this subset was r=0.515)")
    else:
        print(f"  corr(model entropy, H): n/a (zero-variance distributions)")
    order = np.argsort(H)
    t = len(order) // 3
    for name, sl in [("low-H", order[:t]), ("mid-H", order[t:2*t]), ("high-H", order[2*t:])]:
        if len(sl) == 0:
            continue
        print(f"    acc[{name}] = {(ma[sl]==ha[sl]).mean():.3f}  (mean H={H[sl].mean():.3f}, "
              f"mean model entropy={mH[sl].mean():.3f})")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="thinking models burn tokens reasoning before emitting JSON; keep generous")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--key-env", default=DEFAULT_KEY_ENV)
    ap.add_argument("--subset", default=str(SUBSET_PATH),
                    help="path to the item subset (default: 450 stratified, matches Sonnet control)")
    ap.add_argument("--out-tag", default=DEFAULT_OUT_TAG,
                    help="suffix on the output filename so 450 and full runs never collide")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if args.analyze_only:
        analyze(args.model, args.out_tag)
    else:
        run(args.model, args.limit, args.temperature, args.max_tokens,
            args.base_url, args.key_env, args.subset, args.out_tag)


if __name__ == "__main__":
    main()