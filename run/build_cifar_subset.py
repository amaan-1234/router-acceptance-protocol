"""Build a frozen, stratified-by-human-entropy 450-item CIFAR-10H subset.

CIFAR-10H is heavily zero-inflated: most test images are unanimous (H=0), so a random
draw would be ~80% trivial and the "does model uncertainty track human disagreement"
question would be unanswerable. Mirroring the ChaosNLI 450 subset, we stratify by human
entropy and OVERSAMPLE the ambiguous tail, while keeping a representative band of
easy/medium items so accuracy is still meaningful.

Output: outputs/cifar10h_subset_450.json  -- a list of items:
    {idx, human_dist[10], entropy, human_argmax, cifar_label}
The image itself is NOT stored here (arrays are large); the runner re-reads
cifar10_test_images.npy by idx. Frozen + seeded so every model sees identical items,
exactly like the NLI sweep.

Usage:
    python -m run.build_cifar_subset           # writes the 450 subset
    python -m run.build_cifar_subset --n 450 --seed 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from cac import config
from cac.data import cifar10h
from cac.data.labels import CIFAR10_CLASSES

OUT_PATH = config.OUTPUTS_DIR / "cifar10h_subset_450.json"


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def build(n: int, seed: int):
    probs = cifar10h.download_probs()                 # (10000, 10), normalized
    _, labels = cifar10h.download_cifar10_test()      # (10000,), standard order
    rep = cifar10h.alignment_report(probs, labels)
    print(f"[cifar-subset] alignment check: {rep}  (agreement should be ~0.95)")
    assert rep["agreement"] > 0.90, "ordering mismatch between probs and labels!"

    H = np.array([_entropy_bits(probs[i]) for i in range(len(probs))])
    rng = np.random.default_rng(seed)

    # three entropy bands; oversample the ambiguous (high-H) tail
    zero = np.where(H <= 1e-6)[0]            # unanimous
    low = np.where((H > 1e-6) & (H <= 0.6))[0]
    mid = np.where((H > 0.6) & (H <= 1.2))[0]
    high = np.where(H > 1.2)[0]
    print(f"[cifar-subset] pool sizes: zero={len(zero)} low={len(low)} "
          f"mid={len(mid)} high={len(high)}")

    # allocation: keep some easy items for accuracy, weight toward ambiguity
    quota = {"zero": int(0.15 * n), "low": int(0.20 * n),
             "mid": int(0.30 * n), "high": n}  # high fills the remainder
    picks = []
    for band, pool in [("zero", zero), ("low", low), ("mid", mid)]:
        k = min(quota[band], len(pool))
        picks.append(rng.choice(pool, size=k, replace=False))
    used = sum(len(p) for p in picks)
    k_high = min(n - used, len(high))
    picks.append(rng.choice(high, size=k_high, replace=False))
    sel = np.concatenate(picks)
    rng.shuffle(sel)
    sel = sel[:n]

    items = []
    for idx in sel:
        idx = int(idx)
        items.append({
            "idx": idx,
            "human_dist": [round(float(x), 6) for x in probs[idx]],
            "entropy": round(float(H[idx]), 6),
            "human_argmax": int(probs[idx].argmax()),
            "cifar_label": int(labels[idx]),
        })

    json.dump(items, open(OUT_PATH, "w"))
    hh = np.array([it["entropy"] for it in items])
    print(f"[cifar-subset] wrote {len(items)} items -> {OUT_PATH.name}")
    print(f"  entropy: min={hh.min():.3f} mean={hh.mean():.3f} max={hh.max():.3f}  "
          f"(>0: {(hh>1e-6).mean()*100:.0f}% of subset)")
    print(f"  classes (column order): {CIFAR10_CLASSES}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.n, args.seed)


if __name__ == "__main__":
    main()
