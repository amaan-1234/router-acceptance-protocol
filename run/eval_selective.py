"""Selective-prediction evaluation of the real disagreement signal (Phase 2, Option 1).

report.py answers "does disagreement correlate with human uncertainty?" (r).
This answers the Phase 2 question: "does escalating high-disagreement items to a
frontier model actually RECOVER accuracy, and at what budget?" — i.e. is the signal
USEFUL for routing, not just correlated. Run on the SAME real outputs/raw/*.jsonl
report.py reads; no recompute, no dir moves, read-only.

Accuracy target = human-majority label (argmax of the human distribution), matching
run_frontier.py and the cascade_accuracy convention in cac/pipeline/metrics.py.

Two escalation policies compared at every budget b in [0,1]:
  - disagreement-ranked: escalate the top-b fraction by the dual signal
    (alpha*JSD_norm + beta*(1-MTA)); JSD-only if no matching MTA cache)
  - random: escalate a random b fraction (mean over R draws) — the baseline the
    signal must beat to be worth anything

Escalated items are answered by the frontier; non-escalated by the ensemble
majority vote. Frontier accuracy:
  --frontier-acc FLOAT   constant ceiling (default 0.95; the value run_pipeline assumes)
  --frontier-file PATH   per-item real frontier labels (e.g. outputs/frontier_chaosnli.jsonl)

Usage:
  python -m run.eval_selective --dataset cifar10h
  python -m run.eval_selective --dataset chaosnli --frontier-file outputs/frontier_chaosnli.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cac import config
from cac.data import target_source
from cac.ensemble import inference
from cac.ensemble.jsd import mean_pairwise_jsd
from cac.pipeline import weights
from cac.pipeline.metrics import normalise_01
from cac.targets import human_entropy as entropy_of, hard_mask


def _load_mta(n: int) -> np.ndarray | None:
    if config.MTA_SCORES.exists():
        m = np.load(config.MTA_SCORES)
        if len(m) == n:
            return m
    return None


def _frontier_labels(path: str | None, n: int, human_argmax: np.ndarray,
                     acc: float, rng: np.random.Generator):
    """Per-item frontier predictions + optional coverage mask.

    Returns (frontier_pred, covered_mask). With a frontier file, covered_mask marks
    items that have a REAL frontier label; the caller restricts the eval to those so
    uncovered items are never silently scored as correct via a truth fallback. When
    simulating, covered_mask is None (all items covered).
    """
    if path and Path(path).exists():
        fa = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "frontier_argmax" in r and "idx" in r:
                fa[int(r["idx"])] = int(r["frontier_argmax"])
        out = human_argmax.copy()
        covered = np.zeros(n, bool)
        for i in range(n):
            if i in fa:
                out[i] = fa[i]
                covered[i] = True
        print(f"[frontier] real labels for {covered.sum()}/{n} items from {path} "
              f"-> restricting eval to the covered subset")
        return out, covered
    k = int(human_argmax.max()) + 1
    correct = rng.random(n) < acc
    wrong = (human_argmax + rng.integers(1, max(k, 2), size=n)) % max(k, 2)
    return np.where(correct, human_argmax, wrong), None


def selective_curve(signal: np.ndarray, ens_pred: np.ndarray, frontier_pred: np.ndarray,
                    truth: np.ndarray, n_steps: int = 21, n_random: int = 200,
                    seed: int = 0) -> list[dict]:
    n = len(truth)
    rng = np.random.default_rng(seed)
    order = np.argsort(-signal)            # highest disagreement first
    rows = []
    for b in np.linspace(0.0, 1.0, n_steps):
        k = int(round(b * n))
        esc = np.zeros(n, bool)
        esc[order[:k]] = True
        pred = np.where(esc, frontier_pred, ens_pred)
        acc = float((pred == truth).mean())
        # random escalation at the same budget
        racc = []
        for _ in range(n_random):
            rm = np.zeros(n, bool)
            rm[rng.choice(n, size=k, replace=False)] = True
            rp = np.where(rm, frontier_pred, ens_pred)
            racc.append((rp == truth).mean())
        rows.append({"budget": round(float(b), 3),
                     "acc_disagreement": round(acc, 4),
                     "acc_random": round(float(np.mean(racc)), 4)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10h", choices=["cifar10h", "chaosnli"])
    ap.add_argument("--frontier-acc", type=float, default=0.95)
    ap.add_argument("--frontier-file", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    dists, keys = inference.load_distributions()          # (M,N,K) from outputs/raw/
    m, n, k = dists.shape
    human_probs = target_source.human_probs(args.dataset)[:n]
    he = entropy_of(human_probs)
    truly_hard, _, _ = hard_mask(human_probs)
    human_argmax = human_probs.argmax(1)

    # disagreement signal (dual if MTA matches N, else JSD-only — same logic as report.py)
    jsd = mean_pairwise_jsd(dists)
    jsd_norm = normalise_01(jsd)
    mta = _load_mta(n)
    if mta is not None:
        w = weights.mi_weights(jsd_norm, 1.0 - mta, he)
        signal = w["alpha"] * jsd_norm + w["beta"] * (1.0 - mta)
        sig_name = f"dual(alpha={w['alpha']:.2f},beta={w['beta']:.2f})"
    else:
        signal = jsd_norm
        sig_name = "jsd_only"

    ens_pred = dists.mean(0).argmax(1)        # ensemble = mean-distribution argmax
    frontier_pred, covered = _frontier_labels(args.frontier_file, n, human_argmax,
                                              args.frontier_acc, rng)

    # restrict to items with a real frontier label (avoids truth-fallback inflation)
    if covered is not None:
        signal = signal[covered]
        ens_pred = ens_pred[covered]
        frontier_pred = frontier_pred[covered]
        human_argmax = human_argmax[covered]
        n = int(covered.sum())

    ens_acc = float((ens_pred == human_argmax).mean())
    frontier_acc_eff = float((frontier_pred == human_argmax).mean())
    rows = selective_curve(signal, ens_pred, frontier_pred, human_argmax)

    # headline numbers at a few budgets
    by_b = {r["budget"]: r for r in rows}
    bar = "=" * 70
    print(f"\n{bar}\nSELECTIVE-PREDICTION  ({args.dataset}, N={n}, models={keys})\n{bar}")
    print(f"  signal: {sig_name}")
    print(f"  ensemble-only accuracy:  {ens_acc:.4f}")
    print(f"  frontier-on-all accuracy: {frontier_acc_eff:.4f} "
          f"({'real labels' if args.frontier_file else f'simulated @ {args.frontier_acc}'})")
    print(f"  {'budget':>8} {'disagreement':>14} {'random':>10} {'lift':>8}")
    for b in (0.1, 0.25, 0.5):
        r = by_b.get(round(b, 3))
        if r:
            lift = r["acc_disagreement"] - r["acc_random"]
            print(f"  {b:>8.0%} {r['acc_disagreement']:>14.4f} "
                  f"{r['acc_random']:>10.4f} {lift:>+8.4f}")
    # area between disagreement and random curves (overall routing value)
    da = np.array([r["acc_disagreement"] for r in rows])
    ra = np.array([r["acc_random"] for r in rows])
    print(f"  mean lift over random (all budgets): {float((da - ra).mean()):+.4f}")
    print(bar)

    out = Path(args.out or (config.OUTPUTS_DIR / f"selective_{args.dataset}.json"))
    out.write_text(json.dumps({
        "dataset": args.dataset, "n": n, "models": keys, "signal": sig_name,
        "ensemble_acc": ens_acc, "frontier_acc": frontier_acc_eff,
        "frontier_source": args.frontier_file or f"simulated@{args.frontier_acc}",
        "curve": rows}, indent=2))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()