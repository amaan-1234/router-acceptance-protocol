"""Does model code-grading disagreement track human grader disagreement? (Menagerie)

The code-grading analogue of eval_mohler.py, on real 4-rater data. Reads
outputs/menagerie_graders.jsonl and reports, overall and per criterion:

  E1 signal validity : Pearson/Spearman of model disagreement (std across K personas)
                       vs human disagreement (std across the 4 assessors).
  E2 selective grading: rank by model disagreement, escalate top fraction to the
                       adjudicated human grade; accuracy = model within `tol` of
                       adjudicated. Disagreement-ranked vs random at matched budget.
  E3 quartiles       : mean human std within each model-disagreement quartile.

No API calls. Run after the grader ensemble completes (or partway -- it works on
whatever rows exist).

Usage:
    python -m run.eval_menagerie
    python -m run.eval_menagerie --tolerance 1.5 --by-criterion
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

try:
    from cac import config
    DEFAULT_IN = config.OUTPUTS_DIR / "menagerie_graders.jsonl"
    DEFAULT_OUT = config.OUTPUTS_DIR / "menagerie_eval.json"
except Exception:
    DEFAULT_IN = Path("outputs/menagerie_graders.jsonl")
    DEFAULT_OUT = Path("outputs/menagerie_eval.json")


def _load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no records in {path}; run run_menagerie_graders first")
    return rows


def _corr(model_sig, human_sig):
    if np.std(model_sig) < 1e-9 or np.std(human_sig) < 1e-9:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan")}
    return {"pearson_r": round(float(np.corrcoef(model_sig, human_sig)[0, 1]), 4),
            "spearman_rho": round(float(spearmanr(model_sig, human_sig).statistic), 4)}


def _selective(model_sig, model_grade, adjudicated, tol, n_steps=21, n_rand=200, seed=0):
    n = len(adjudicated)
    rng = np.random.default_rng(seed)
    order = np.argsort(-model_sig)
    model_correct = np.abs(model_grade - adjudicated) <= tol
    rows = []
    for b in np.linspace(0, 1, n_steps):
        k = int(round(b * n))
        esc = np.zeros(n, bool); esc[order[:k]] = True
        acc = float(np.where(esc, True, model_correct).mean())
        racc = []
        for _ in range(n_rand):
            rm = np.zeros(n, bool); rm[rng.choice(n, k, replace=False)] = True
            racc.append(np.where(rm, True, model_correct).mean())
        rows.append({"budget": round(float(b), 3), "acc_disagreement": round(acc, 4),
                     "acc_random": round(float(np.mean(racc)), 4)})
    return rows


def _quartiles(model_sig, human_sig):
    q = np.quantile(model_sig, [0.25, 0.5, 0.75])
    b = np.digitize(model_sig, q)
    return {f"Q{i+1}": round(float(human_sig[b == i].mean()), 4)
            for i in range(4) if (b == i).any()}


def _block(rows, tol, label):
    model_sig = np.array([r["model_std"] for r in rows], float)
    model_grade = np.array([r["model_mean"] for r in rows], float)
    human_sig = np.array([r["human_std"] for r in rows], float)
    adjudicated = np.array([r["human_mean"] for r in rows], float)
    e1 = _corr(model_sig, human_sig)
    e3 = _quartiles(model_sig, human_sig)
    e2 = _selective(model_sig, model_grade, adjudicated, tol)
    model_acc = float((np.abs(model_grade - adjudicated) <= tol).mean())
    by_b = {r["budget"]: r for r in e2}
    da = np.array([r["acc_disagreement"] for r in e2]); ra = np.array([r["acc_random"] for r in e2])
    print(f"\n--- {label} (N={len(rows)}) ---")
    print(f"  E1 model-disagree vs human-disagree: r={e1['pearson_r']} rho={e1['spearman_rho']}")
    print(f"  E3 mean human std by model-disagree quartile: {e3}")
    print(f"  model accuracy (within {tol} of adjudicated): {model_acc:.4f}")
    for bb in (0.1, 0.25, 0.5):
        r = by_b.get(round(bb, 3))
        if r:
            print(f"  E2 @ {bb:.0%}: disagree={r['acc_disagreement']:.4f} "
                  f"random={r['acc_random']:.4f} lift={r['acc_disagreement']-r['acc_random']:+.4f}")
    print(f"  E2 mean lift over random: {float((da-ra).mean()):+.4f}")
    return {"n": len(rows), "E1": e1, "E3_quartile": e3,
            "model_accuracy": round(model_acc, 4), "E2_curve": e2,
            "E2_mean_lift": round(float((da - ra).mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--tolerance", type=float, default=1.5,
                    help="model grade counts correct if within this of adjudicated (0-13 scale)")
    ap.add_argument("--by-criterion", action="store_true")
    args = ap.parse_args()

    rows = _load(args.in_file)
    print("=" * 70)
    print(f"MENAGERIE CODE-GRADING DISAGREEMENT  (signal=model_std, tol={args.tolerance})")
    print("=" * 70)
    report = {"overall": _block(rows, args.tolerance, "OVERALL (all criteria)")}

    if args.by_criterion:
        for crit in sorted(set(r["criterion"] for r in rows)):
            sub = [r for r in rows if r["criterion"] == crit]
            if len(sub) >= 8:
                report[crit] = _block(sub, args.tolerance, crit)

    print("=" * 70)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
