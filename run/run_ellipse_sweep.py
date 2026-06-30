"""Cross-family grading sweep over ELLIPSE essays (Option B: point scores).

ONE MODEL PER JOB. Each panel model gives ONE Overall point score (1.0-5.0, 0.5
steps) per essay. Cross-MODEL spread of those scores = the routing signal,
computed at eval from the per-model jsonls. Both human targets emitted per record.
Output: outputs/ellipse_sweep_<model>_<tag>.jsonl
"""
from __future__ import annotations
import argparse, json, os, re, time
from pathlib import Path
import numpy as np
try:
    from cac import config
    OUT_DIR = Path(config.OUTPUTS_DIR)
except Exception:
    OUT_DIR = Path("outputs")
from cac.data import ellipse  # type: ignore

BINS = [round(1.0 + 0.5 * i, 1) for i in range(9)]
NBINS = len(BINS)
BIN_INDEX = {b: i for i, b in enumerate(BINS)}
SCORE_MIN, SCORE_MAX = 1.0, 5.0
PROMPT = (
    "You are scoring an essay written by an English Language Learner (grades 8-12) "
    "for OVERALL holistic language proficiency on a scale from 1.0 (lowest) to "
    "5.0 (highest) in increments of 0.5. Consider cohesion, syntax, vocabulary, "
    "phraseology, grammar, and conventions together as a single overall judgment.\n\n"
    "Essay:\n{text}\n\n"
    'Output ONLY a JSON object with your single best score, e.g. {{"score": 3.5}}. Nothing else.'
)

def _client(base_url, key_env):
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai --break-system-packages")
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"{key_env} not set. export {key_env}=...")
    return OpenAI(base_url=base_url, api_key=key)

def _entropy_bits(p):
    p = np.asarray(p, dtype=float); p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0

def _parse_score(text):
    if not text: return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{": depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:j+1])
                        if "score" in obj:
                            return max(SCORE_MIN, min(SCORE_MAX, float(obj["score"])))
                    except Exception: pass
                    break
        start = text.find("{", start+1)
    for tok in re.findall(r"\d+(?:\.\d+)?", text):
        try: v = float(tok)
        except ValueError: continue
        if SCORE_MIN <= v <= SCORE_MAX: return v
    return None

def _call(client, model, text, temperature, max_tokens, max_retries=2):
    msg = PROMPT.format(text=text)
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, max_tokens=max_tokens,
                          messages=[{"role": "user", "content": msg}])
            if temperature is not None:
                kwargs["temperature"] = temperature + 0.2 * attempt
            resp = client.chat.completions.create(**kwargs)
            m = resp.choices[0].message
            raw = (m.content or getattr(m, "reasoning_content", None) or "")
            s = _parse_score(raw)
            if s is not None:
                return s, raw, True
            if attempt < max_retries - 1:
                continue
            return None, raw, False
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [retry {attempt+1}/{max_retries} in {wait}s] {type(e).__name__}: {str(e)[:100]}")
            time.sleep(wait)
    return None, "", False

def _human_dist(o1, o2):
    d = np.zeros(NBINS)
    for s in (round(o1,1), round(o2,1)):
        if s in BIN_INDEX: d[BIN_INDEX[s]] += 0.5
    return d/d.sum() if d.sum() else np.ones(NBINS)/NBINS

def _load_done(out_path):
    if not out_path.exists(): return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try: done.add(json.loads(line)["uid"])
            except Exception: continue
    return done

def run(model, limit, temperature, full, base_url, key_env, max_tokens, out_tag):
    items = ellipse.load(drop_identifying=True)
    if not full:
        items = ellipse.stratified_subset(items, n=450)
    if limit:
        items = items[:limit]
    out_path = OUT_DIR / f"ellipse_sweep_{model}_{out_tag}.jsonl"
    done = _load_done(out_path)
    client = _client(base_url, key_env)
    print(f"[ellipse-sweep] model={model} items={len(items)} "
          f"(already done: {len(done)}) temp={temperature} max_tokens={max_tokens} -> {out_path.name}")
    with open(out_path, "a", encoding="utf-8") as f:
        for n, it in enumerate(items, 1):
            if it.uid in done: continue
            score, raw, ok = _call(client, model, it.text, temperature, max_tokens)
            hd = _human_dist(it.overall_1, it.overall_2)
            rec = {
                "uid": it.uid, "model": model,
                "model_score": (round(float(score), 2) if score is not None else None),
                "parse_ok": ok, "raw": raw,
                "overall_1": it.overall_1, "overall_2": it.overall_2,
                "rater_gap": it.rater_gap, "human_grade": it.human_grade,
                "human_dist": [round(float(x), 4) for x in hd],
                "human_entropy": round(_entropy_bits(hd), 4),
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            if n % 25 == 0:
                print(f"  [{n}/{len(items)}] uid={it.uid[:10]} score={rec['model_score']} ok={ok} rater_gap={it.rater_gap}")
    print(f"[ellipse-sweep] done -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--key-env", default="LOCAL_API_KEY")
    ap.add_argument("--out-tag", default="sub450")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    temp = None if args.temperature is None else args.temperature
    run(args.model, args.limit, temp, args.full, args.base_url,
        args.key_env, args.max_tokens, args.out_tag)

if __name__ == "__main__":
    main()
