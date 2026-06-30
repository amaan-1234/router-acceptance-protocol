"""Sweep eval over all per-model ELLIPSE grader files (ellipse_graders_<model>_<tag>.jsonl).
Reuses eval_ellipse's E1/E2/E3 functions; prints a per-model table. No API calls.
Usage: python -m run.eval_ellipse_sweep --tag sub450 --signal model_gap --tolerance 0.5
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from run.eval_ellipse import e1_validity, e2_selective, e3_quartiles
try:
    from cac import config
    OUT = str(config.OUTPUTS_DIR)
except Exception:
    OUT = "outputs"


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sub450")
    ap.add_argument("--signal", default="model_gap", choices=["model_gap", "model_std"])
    ap.add_argument("--tolerance", type=float, default=0.5)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(OUT, f"ellipse_graders_*_{args.tag}.jsonl")))
    if not files:
        raise SystemExit(f"no ellipse_graders_*_{args.tag}.jsonl in {OUT}")

    print(f"signal={args.signal}  tol=±{args.tolerance}  tag={args.tag}\n")
    print(f"{'model':22s} {'N':>4} {'pearson':>8} {'spear':>7} "
          f"{'sel@10':>7} {'sel@25':>7} {'rand@25':>8}")
    print("-" * 70)
    rows_out = {}
    for f in files:
        name = os.path.basename(f)[len("ellipse_graders_"):-len(f"_{args.tag}.jsonl")]
        rows = _load(f)
        sig = np.array([r[args.signal] for r in rows], float)
        grade = np.array([r["model_mean"] for r in rows], float)
        adj = np.array([r["human_grade"] for r in rows], float)
        rgap = np.array([r["rater_gap"] for r in rows], float)
        v = e1_validity(sig, rgap)
        e2 = e2_selective(sig, grade, adj, args.tolerance)
        byb = {round(r["budget"], 2): r for r in e2}

        def acc(b, key):
            k = min(byb, key=lambda x: abs(x - b))
            return byb[k][key]
        print(f"{name:22s} {len(rows):>4} {v['pearson_r']:>8.3f} {v['spearman_rho']:>7.3f} "
              f"{acc(0.10,'acc_disagreement'):>7.3f} {acc(0.25,'acc_disagreement'):>7.3f} "
              f"{acc(0.25,'acc_random'):>8.3f}")
        rows_out[name] = {"N": len(rows), **v}
    json.dump(rows_out, open(os.path.join(OUT, f"ellipse_eval_sweep_{args.tag}.json"), "w"), indent=2)
    print(f"\nwrote {OUT}/ellipse_eval_sweep_{args.tag}.json")


if __name__ == "__main__":
    main()
