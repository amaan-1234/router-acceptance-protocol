"""A1 selective-routing eval over existing sweep .jsonl ensembles (no re-serving).
Reuses eval_selective.selective_curve + cac JSD. Aligns ensemble/frontier on `idx`.

Task 2 (A1 on ChaosNLI):
  python -m run.eval_selective_sweep --prefix sweep_chaosnli_ --tag sub450 \
      --frontier-file outputs/frontier_chaosnli_450.jsonl --label NLI-text
Task 3 (modality specialists):
  python -m run.eval_selective_sweep --prefix vision_cifar_ --tag sub450 --label VISION
  python -m run.eval_selective_sweep --prefix sweep_chaosnli_ --tag sub450 \
      --frontier-file outputs/frontier_chaosnli_450.jsonl --label TEXT
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from run.eval_selective import selective_curve
from cac.ensemble.jsd import mean_pairwise_jsd
from cac.pipeline.metrics import normalise_01
try:
    from cac import config; OUT = str(config.OUTPUTS_DIR)
except Exception:
    OUT = "outputs"


def _read(path):
    return {json.loads(l)["idx"]: json.loads(l)
            for l in open(path, encoding="utf-8") if l.strip()}


def _aurc(rows):  # area under risk-coverage of the disagreement policy (lower=better)
    cov = np.array([1 - r["budget"] for r in rows])     # coverage = kept (non-escalated)
    risk = 1 - np.array([r["acc_disagreement"] for r in rows])
    o = np.argsort(cov)
    return float(np.trapezoid(risk[o], cov[o]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tag", default="sub450")
    ap.add_argument("--frontier-file", default=None)
    ap.add_argument("--label", default="ENSEMBLE")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(OUT, f"{args.prefix}*_{args.tag}.jsonl"))
                   if "frontier" not in os.path.basename(f))
    if len(files) < 2:
        raise SystemExit(f"need >=2 model files for {args.prefix}*_{args.tag}; found {len(files)}")

    per_model = [_read(f) for f in files]
    idxs = sorted(set.intersection(*[set(d) for d in per_model]))
    K = len(per_model[0][idxs[0]]["model_dist"])
    dists = np.array([[m[i]["model_dist"] for i in idxs] for m in per_model], float)  # (M,N,K)
    human_argmax = np.array([per_model[0][i]["human_argmax"] for i in idxs])
    ens_pred = dists.mean(0).argmax(1)
    signal = normalise_01(mean_pairwise_jsd(dists))

    # frontier labels (real file -> restrict to covered subset; else frontier=human ceiling)
    if args.frontier_file and os.path.exists(args.frontier_file):
        fr = _read(args.frontier_file)
        covered = np.array([i in fr for i in idxs])
        frontier_pred = np.array([fr[i]["frontier_argmax"] if i in fr else human_argmax[j]
                                  for j, i in enumerate(idxs)])
        signal, ens_pred = signal[covered], ens_pred[covered]
        frontier_pred, human_argmax = frontier_pred[covered], human_argmax[covered]
        src = f"real frontier {covered.sum()}/{len(idxs)}"
    else:
        frontier_pred = human_argmax.copy(); src = "frontier=human ceiling (no file)"

    rows = selective_curve(signal, ens_pred, frontier_pred, human_argmax)
    byb = {r["budget"]: r for r in rows}
    ens_acc = float((ens_pred == human_argmax).mean())
    fr_acc = float((frontier_pred == human_argmax).mean())
    da = np.array([r["acc_disagreement"] for r in rows])
    ra = np.array([r["acc_random"] for r in rows])

    print(f"\n===== A1 SELECTIVE [{args.label}] =====")
    print(f"models={len(files)} N={len(human_argmax)} ({src})")
    print(f"ensemble-acc={ens_acc:.4f}  frontier-acc={fr_acc:.4f}  AURC={_aurc(rows):.4f}")
    print(f"{'budget':>7} {'disagree':>9} {'random':>8} {'lift':>8}")
    for b in (0.10, 0.25, 0.50):
        r = byb.get(round(b, 3))
        if r: print(f"{b:>7.0%} {r['acc_disagreement']:>9.4f} {r['acc_random']:>8.4f} {r['acc_disagreement']-r['acc_random']:>+8.4f}")
    print(f"mean lift over random (all budgets): {float((da-ra).mean()):+.4f}")


if __name__ == "__main__":
    main()
