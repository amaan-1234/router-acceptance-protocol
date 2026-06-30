"""Vision capacity-sweep runner: stated 10-class distribution per model on the frozen
CIFAR-10H 450 subset, via any OpenAI-compatible endpoint (a hosted endpoint or local vLLM).

The image-track analog of run_ensemble_api.py. For each CIFAR-10H item it sends the
32x32 image (base64 PNG) and asks ONE model for its probability distribution over the
10 CIFAR classes, in a single call. Output schema MATCHES the NLI runner, so
eval_sweep.py reads these files unchanged (just a different prefix).

Same two endpoint-specific fixes as the text runner:
  1. reads `content` OR `reasoning_content` (thinking models put output in the latter)
  2. brace-scanning parser tolerates reasoning traces full of stray braces

Usage:
    export LLM_API_KEY=...
    python -m run.build_cifar_subset                              # once: build the subset
    python -m run.run_vision_api --model qwen35-0p8b --limit 5    # smoke test
    python -m run.run_vision_api --model qwen35-0p8b              # full 450
    # local models: serve via vLLM, then
    python -m run.run_vision_api --model qwen35-2b \
        --base-url http://localhost:8000/v1 --key-env LOCAL_API_KEY

Caching: appends to outputs/vision_cifar_<model>_sub450.jsonl, flushed per item;
restart skips done idx -> no re-querying.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from cac import config
from cac.data import cifar10h
from cac.data.labels import CIFAR10_CLASSES

SUBSET_PATH = config.OUTPUTS_DIR / "cifar10h_subset_450.json"
DEFAULT_OUT_TAG = "sub450"
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_KEY_ENV = "LLM_API_KEY"

PROMPT = (
    "You are an expert image classifier. Classify this 32x32 image into exactly one of "
    "these 10 CIFAR-10 categories: airplane, automobile, bird, cat, deer, dog, frog, "
    "horse, ship, truck. Images can be ambiguous, so express your judgment as a "
    "probability distribution over the 10 categories rather than a single answer.\n\n"
    "Output ONLY a JSON object with all 10 category names as keys and probabilities that "
    'sum to 1.0, e.g. {"airplane": 0.7, "automobile": 0.1, "bird": 0.0, "cat": 0.0, '
    '"deer": 0.0, "dog": 0.0, "frog": 0.0, "horse": 0.0, "ship": 0.1, "truck": 0.1}. '
    "If the image is ambiguous spread the mass; if clear concentrate it. "
    "End your reply with the JSON object."
)


def _safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")


def _out_path(model: str, tag: str = DEFAULT_OUT_TAG):
    return config.OUTPUTS_DIR / f"vision_cifar_{_safe_name(model)}_{tag}.jsonl"


def _client(base_url: str, key_env: str):
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai --break-system-packages")
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"{key_env} not set. export {key_env}=... (then re-export so "
                         f"Python sees it)")
    return OpenAI(base_url=base_url, api_key=key)


def _img_b64(arr: np.ndarray) -> str:
    """uint8 (32,32,3) -> base64 PNG. Upscale 4x so tiny images survive vision encoders."""
    from PIL import Image
    im = Image.fromarray(arr, "RGB").resize((128, 128), Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _parse_dist(text: str) -> np.ndarray | None:
    """Lenient parse -> (10,) prob vector. Tolerates code fences, [..] used in place of
    {..}, and missing class keys (filled with 0). Returns the LAST viable object."""
    if not text:
        return None
    # drop code fences so ```json blocks don't confuse delimiter matching
    t = text.replace("```json", "").replace("```", "")
    # candidate objects: both { } and [ ] wrappers (some models emit "key":v inside [])
    cands = re.findall(r"[\{\[][^\{\}\[\]]*[\}\]]", t, re.DOTALL)
    for cand in reversed(cands):
        body = cand.strip().strip("[]{}")
        pairs = re.findall(r'"([a-z_]+)"\s*:\s*([0-9]*\.?[0-9]+)', body)
        if not pairs:
            continue
        d = {k: float(v) for k, v in pairs}
        present = sum(1 for c in CIFAR10_CLASSES if c in d)
        if present < 6:
            continue
        v = np.array([d.get(c, 0.0) for c in CIFAR10_CLASSES], dtype=np.float64)
        s = v.sum()
        if np.isfinite(s) and s > 0:
            return v / s
    # FALLBACK: prose/markdown "class: prob" anywhere in the trace (handles reasoning
    # traces that state a partial distribution without a closing JSON object). Take the
    # LAST stated value per class; accept >=3 classes with mass; renormalize.
    last = {}
    for m in re.finditer(r'(airplane|automobile|bird|cat|deer|dog|frog|horse|ship|truck)\b[^0-9\n]{0,12}([01]?\.[0-9]+)', t, re.I):
        last[m.group(1).lower()] = float(m.group(2))
    last = {k: v for k, v in last.items() if v > 0}
    if len(last) >= 3:
        v = np.array([last.get(c, 0.0) for c in CIFAR10_CLASSES], dtype=np.float64)
        s = v.sum()
        if np.isfinite(s) and s > 0:
            return v / s
    return None


def _call(client, model, img_b64, temperature, max_tokens, max_retries=2):
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        {"type": "text", "text": PROMPT},
    ]
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, max_tokens=max_tokens,
                          messages=[{"role": "user", "content": content}])
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            ch = resp.choices[0]
            m = ch.message
            text = (m.content or getattr(m, "reasoning_content", None) or "")
            dist = _parse_dist(text)
            # thinking models can run past the budget; if truncated AND unparseable, retry with more room
            if dist is None and attempt < max_retries - 1:
                fr = getattr(ch, "finish_reason", None)
                # reasoning doom-loop: more tokens feeds it. Nudge temperature to break determinism.
                kwargs["temperature"] = 0.5
                print(f"    [unparseable (finish={fr}) -> retry {attempt+1} at temp=0.5]")
                continue
            return dist, text
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
                done.add(json.loads(line)["idx"])
            except Exception:
                continue
    return done


def run(model, limit, temperature, max_tokens, base_url, key_env, out_tag):
    if not SUBSET_PATH.exists():
        raise SystemExit(f"missing {SUBSET_PATH}; run: python -m run.build_cifar_subset")
    items = json.load(open(SUBSET_PATH, encoding="utf-8"))
    if limit:
        items = items[:limit]
    images, _ = cifar10h.download_cifar10_test()   # (10000,32,32,3), indexed by idx
    out_path = _out_path(model, out_tag)
    done = _load_done(out_path)
    client = _client(base_url, key_env)
    print(f"[vision] model={model} subset={SUBSET_PATH.name} base={base_url} "
          f"temp={temperature} max_tokens={max_tokens} "
          f"items={len(items)} (already done: {len(done)}) -> {out_path.name}")

    K = len(CIFAR10_CLASSES)
    with open(out_path, "a", encoding="utf-8") as f:
        for n_done, it in enumerate(items, 1):
            if it["idx"] in done:
                continue
            b64 = _img_b64(images[it["idx"]])
            dist, raw = _call(client, model, b64, temperature, max_tokens)
            if dist is None:
                dist = np.full(K, 1.0 / K); parse_ok = False
            else:
                parse_ok = True
            rec = {
                "uid": str(it["idx"]), "idx": it["idx"], "model": model,
                "model_dist": [round(float(x), 4) for x in dist],
                "model_argmax": int(dist.argmax()),
                "model_entropy_bits": round(_entropy_bits(dist), 4),
                "parse_ok": parse_ok,
                "raw": raw,
                "human_dist": it["human_dist"],
                "human_entropy": it["entropy"],
                "human_argmax": int(it["human_argmax"]),
            }
            f.write(json.dumps(rec) + "\n"); f.flush()
            if n_done % 20 == 0:
                print(f"  [{n_done}/{len(items)}] idx={it['idx']} argmax={rec['model_argmax']} ok={parse_ok}")
    analyze(model, out_tag)


def analyze(model, out_tag=DEFAULT_OUT_TAG):
    out_path = _out_path(model, out_tag)
    recs = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    if not recs:
        print("[vision] no results yet."); return
    bad = sum(1 for r in recs if not r.get("parse_ok", True))
    ma = np.array([r["model_argmax"] for r in recs])
    ha = np.array([r["human_argmax"] for r in recs])
    H = np.array([r["human_entropy"] for r in recs])
    mH = np.array([r["model_entropy_bits"] for r in recs])

    print("\n" + "=" * 60)
    print(f"VISION SWEEP POINT  model={model}  (N={len(recs)}, parse failures={bad})")
    print("=" * 60)
    print(f"  accuracy vs human-majority        : {(ma==ha).mean():.3f}")
    if mH.std() > 1e-9:
        r = float(np.corrcoef(mH, H)[0, 1])
        print(f"  corr(model entropy, H_human)      : {r:.3f}")
    else:
        print(f"  corr(model entropy, H): n/a (zero-variance)")
    order = np.argsort(H)
    t = len(order) // 3
    for name, sl in [("low-H", order[:t]), ("mid-H", order[t:2*t]), ("high-H", order[2*t:])]:
        if len(sl):
            print(f"    acc[{name}] = {(ma[sl]==ha[sl]).mean():.3f}  (mean H={H[sl].mean():.3f}, "
                  f"mean model entropy={mH[sl].mean():.3f})")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--key-env", default=DEFAULT_KEY_ENV)
    ap.add_argument("--out-tag", default=DEFAULT_OUT_TAG)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if args.analyze_only:
        analyze(args.model, args.out_tag)
    else:
        run(args.model, args.limit, args.temperature, args.max_tokens,
            args.base_url, args.key_env, args.out_tag)


if __name__ == "__main__":
    main()