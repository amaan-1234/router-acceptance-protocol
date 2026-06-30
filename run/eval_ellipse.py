"""Does model grading-disagreement track instructor disagreement? (ELLIPSE, Phase 2)

The grading analogue of run/eval_mohler.py, on the ELLIPSE essay corpus.
Reads outputs/ellipse_graders.jsonl (from run_ellipse_graders) and reports:

  E1 signal validity : Pearson/Spearman of model disagreement (std or gap across the
                       K grader personas) vs instructor disagreement (|Overall_1-Overall_2|).

  E2 selective grading: rank items by model disagreement, "escalate" the top fraction
                       to the adjudicated grade (instructor mean), rest graded by the
                       model panel mean. Accuracy = within-tolerance of adjudicated.
                       Disagreement-ranked escalation vs random at matched budget.

  E3 calibration check: mean instructor gap within each model-disagreement quartile.

No API calls -- pure post-processing. Run after the grader ensemble finishes.

Usage:
    python -m run.eval_ellipse
    python -m run.eval_ellipse --tolerance 0.5 --signal model_std
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

try:
    from cac import config
    DEFAULT_IN = config.OUTPUTS_DIR / "ellipse_graders.jsonl"
    DEFAULT_OUT = config.OUTPUTS_DIR / "ellipse_eval.json"
except Exception:
    DEFAULT_IN = Path("outputs/ellipse_graders.jsonl")
    DEFAULT_OUT = Path("outputs/ellipse_eval.json")


def _load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no records in {path}; run run_ellipse_graders first")
    return rows


def e1_validity(model_sig, rater_gap):
    r = float(np.corrcoef(model_sig, rater_gap)[0, 1])
    rho = float(spearmanr(model_sig, rater_gap).statistic)
    return {"pearson_r": round(r, 4), "spearman_rho": round(rho, 4)}


def e2_selective(model_sig, model_grade, adjudicated, tol, n_steps=21, n_rand=200, seed=0):
    n = len(adjudicated)
    rng = np.random.default_rng(seed)
    order = np.argsort(-model_sig)
    model_correct = np.abs(model_grade - adjudicated) <= tol
    rows = []
    for b in np.linspace(0, 1, n_steps):
        k = int(round(b * n))
        esc = np.zeros(n, bool); esc[order[:k]] = True
        correct = np.where(esc, True, model_correct)
        acc = float(correct.mean())
        racc = []
        for _ in range(n_rand):
            rm = np.zeros(n, bool); rm[rng.choice(n, k, replace=False)] = True
            racc.append(np.where(rm, True, model_correct).mean())
        rows.append({"budget": round(float(b), 3),
                     "acc_disagreement": round(acc, 4),
                     "acc_random": round(float(np.mean(racc)), 4)})
    return rows


def e3_quartiles(model_sig, rater_gap):
    q = np.quantile(model_sig, [0.25, 0.5, 0.75])
    bins = np.digitize(model_sig, q)
    return {f"Q{b+1}": round(float(rater_gap[bins == b].mean()), 4)
            for b in range(4) if (bins == b).any()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--signal", default="model_std", choices=["model_std", "model_gap"])
    ap.add_argument("--tolerance", type=float, default=0.5,
                    help="model grade counts correct if within this of adjudicated")
    args = ap.parse_args()

    rows = _load(args.in_file)
    model_sig = np.array([r[args.signal] for r in rows], dtype=float)
    model_grade = np.array([r["model_mean"] for r in rows], dtype=float)
    rater_gap = np.array([r["rater_gap"] for r in rows], dtype=float)
    adjudicated = np.array([r["human_grade"] for r in rows], dtype=float)

    e1 = e1_validity(model_sig, rater_gap)
    e3 = e3_quartiles(model_sig, rater_gap)
    e2 = e2_selective(model_sig, model_grade, adjudicated, args.tolerance)
    model_acc = float((np.abs(model_grade - adjudicated) <= args.tolerance).mean())

    bar = "=" * 70
    print(f"\n{bar}\nELLIPSE GRADING DISAGREEMENT  (N={len(rows)}, signal={args.signal})\n{bar}")
    print(f"  E1 model-disagreement vs instructor-gap: "
          f"r={e1['pearson_r']}  rho={e1['spearman_rho']}")
    print(f"  E3 mean instructor gap by model-disagreement quartile: {e3}")
    print(f"  model grade accuracy (within {args.tolerance} of adjudicated): {model_acc:.4f}")
    by_b = {r["budget"]: r for r in e2}
    print(f"  E2 selective grading (escalate high-model-disagreement -> instructor):")
    print(f"     {'budget':>8} {'disagreement':>14} {'random':>10} {'lift':>8}")
    for b in (0.1, 0.25, 0.5):
        r = by_b.get(round(b, 3))
        if r:
            print(f"     {b:>8.0%} {r['acc_disagreement']:>14.4f} {r['acc_random']:>10.4f} "
                  f"{r['acc_disagreement']-r['acc_random']:>+8.4f}")
    da = np.array([r["acc_disagreement"] for r in e2]); ra = np.array([r["acc_random"] for r in e2])
    print(f"     mean lift over random: {float((da-ra).mean()):+.4f}")
    print(bar)

    Path(args.out).write_text(json.dumps(
        {"n": len(rows), "signal": args.signal, "tolerance": args.tolerance,
         "E1": e1, "E3_quartile_gap": e3, "model_accuracy": round(model_acc, 4),
         "E2_curve": e2}, indent=2))
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
