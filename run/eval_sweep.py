"""Capacity-sweep evaluator for the ChaosNLI uncertainty curve. No API calls.

Reads every outputs/sweep_chaosnli_*.jsonl (and optionally the Sonnet frontier file)
and reports, per model AND as a size-ordered curve:

  * accuracy vs human majority
  * Pearson r(model entropy, H_human)      -- the old headline; demoted to one column
  * Spearman rho(model entropy, H_human)   -- rank version, robust to the zero-inflated H
  * AUROC: does model entropy rank HARD items (human entropy > median) above easy ones?
  * Risk-coverage: accept the most-confident (lowest-entropy) coverage fraction; report
    selective accuracy at 50/70/90% coverage and AURC (area under the risk-coverage
    curve). Random-selection baseline included so the lift is visible.

Then two cross-model artifacts the colleague asked for:
  * r-vs-size CURVE: Pearson r and AUROC traced against model parameter count, so the
    "small models can't track human disagreement" claim becomes a curve, not a 2-point
    contrast. MoE models report (total / active) params; the active count is flagged.
  * PAIRWISE ERROR-CORRELATION: phi between the error-indicator vectors of each model
    pair on shared items -- tests "confidently wrong in UNCORRELATED ways". Low
    off-diagonal correlation = errors are idiosyncratic (the claim); high = shared.

Usage:
    python -m run.eval_sweep
    python -m run.eval_sweep --include-frontier      # add Sonnet as the top curve point
    python -m run.eval_sweep --hard-quantile 0.5     # threshold defining a "hard" item
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

try:
    from cac import config
    OUT_DIR = Path(config.OUTPUTS_DIR)
except Exception:
    OUT_DIR = Path("outputs")

# Parameter sizes (billions). MoE: total params; active params noted separately.
# Curve is ordered by ACTIVE params for the capacity axis (what actually computes),
# with total shown for context. Edit if the deployment differs.
SIZES_TOTAL = {
    "qwen35_0p8b": 0.8, "qwen35_2b": 2.0, "qwen35_4b": 4.0, "qwen35_9b": 9.0,
    "qwen35_27b_fp8": 27.0, "qwen35_122b_a10b_fp8": 122.0,
    "gemma4_e2b_it": 2.0, "gemma4_31b_it": 31.0,
}
SIZES_ACTIVE = {  # only differs for MoE
    "qwen35_122b_a10b_fp8": 10.0,
}


def _size_lookup(name):
    """Match 'qwen35_2b_sub450' -> SIZES key 'qwen35_2b' by longest-prefix match."""
    keys = sorted([k for k in SIZES_TOTAL if name.startswith(k)], key=len, reverse=True)
    if not keys:
        return None, None
    k = keys[0]
    return SIZES_TOTAL[k], SIZES_ACTIVE.get(k, SIZES_TOTAL[k])


def _read(path: Path):
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = []
    for r in recs:
        # accept both the sweep schema and the frontier schema
        ent = r.get("model_entropy_bits", r.get("frontier_entropy_bits"))
        amax = r.get("model_argmax", r.get("frontier_argmax"))
        if ent is None or amax is None:
            continue
        out.append({
            "uid": r["uid"],
            "ent": float(ent),
            "amax": int(amax),
            "H": float(r["human_entropy"]),
            "hmax": int(r["human_argmax"]),
            "ok": bool(r.get("parse_ok", True)),
        })
    return out


def _auroc(score, label):
    """AUROC via Mann-Whitney U. score: higher = predicted positive. label: bool array."""
    pos = score[label]; neg = score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])   # positives occupy the first len(pos) entries
    # tie-safe average ranks
    sort_idx = np.argsort(allv, kind="mergesort")
    sorted_v = allv[sort_idx]
    avg_ranks = np.empty(len(allv), dtype=np.float64)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j+1] == sorted_v[i]:
            j += 1
        avg_ranks[sort_idx[i:j+1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rsum = avg_ranks[:len(pos)].sum()   # positives are the first len(pos) entries of allv
    u = rsum - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _risk_coverage(conf, correct, cov_points=(0.5, 0.7, 0.9)):
    """conf: higher = more confident (use -entropy). Accept top-coverage by conf."""
    n = len(conf)
    order = np.argsort(-conf)  # most confident first
    correct_sorted = correct[order]
    # AURC over a fine grid
    covs = np.linspace(1.0/n, 1.0, n)
    sel_acc = np.cumsum(correct_sorted) / np.arange(1, n + 1)
    aurc = float((np.trapezoid if hasattr(np,"trapezoid") else np.trapz)(1.0 - sel_acc, covs))  # area under risk(=error) vs coverage
    at = {}
    for c in cov_points:
        k = max(1, int(round(c * n)))
        at[c] = float(correct_sorted[:k].mean())
    rand = float(correct.mean())  # random selection accuracy = base accuracy at any coverage
    return aurc, at, rand


def evaluate(include_frontier, hard_quantile, prefix="sweep_chaosnli_", frontier_file="frontier_chaosnli.jsonl"):
    files = sorted(glob.glob(str(OUT_DIR / f"{prefix}*.jsonl")))
    paths = {Path(f).stem.replace(prefix, ""): Path(f) for f in files}
    if include_frontier:
        fp = OUT_DIR / frontier_file
        if fp.exists():
            paths["frontier_sonnet"] = fp
        else:
            print(f"[note] include-frontier set but {frontier_file} not found; skipping frontier point")

    if not paths:
        raise SystemExit("no sweep_chaosnli_*.jsonl files in outputs/")

    per_model = {}
    err_vecs = {}   # uid -> error indicator, for pairwise correlation
    for name, path in paths.items():
        recs = _read(path)
        if len(recs) < 10:
            print(f"[skip] {name}: only {len(recs)} records"); continue
        ent = np.array([r["ent"] for r in recs])
        amax = np.array([r["amax"] for r in recs])
        H = np.array([r["H"] for r in recs])
        hmax = np.array([r["hmax"] for r in recs])
        bad = sum(1 for r in recs if not r["ok"])

        acc = float((amax == hmax).mean())
        pear = float(np.corrcoef(ent, H)[0, 1]) if ent.std() > 1e-9 else float("nan")
        spear = float(spearmanr(ent, H).statistic) if ent.std() > 1e-9 else float("nan")

        thr = np.quantile(H, hard_quantile)
        hard = H > thr
        auroc = _auroc(ent, hard)

        correct = (amax == hmax).astype(np.float64)
        aurc, at, rand = _risk_coverage(-ent, correct)

        _tot, _act = _size_lookup(name)
        per_model[name] = dict(n=len(recs), bad=bad, acc=acc, pearson=pear, spearman=spear,
                               auroc=auroc, aurc=aurc, sel=at, rand=rand,
                               total=_tot, active=_act)
        err_vecs[name] = {r["uid"]: 1.0 - (r["amax"] == r["hmax"]) for r in recs}

    # ---- per-model table ----
    print("\n" + "=" * 96)
    print(f"{'model':<26}{'N':>5}{'acc':>7}{'pearson':>9}{'spear':>8}{'AUROC':>7}"
          f"{'AURC':>7}{'sel@70':>8}{'rand':>7}")
    print("-" * 96)
    def _sortkey(kv):
        a = kv[1]["active"]
        return (a is None, a if a is not None else 0)
    for name, m in sorted(per_model.items(), key=_sortkey):
        print(f"{name:<26}{m['n']:>5}{m['acc']:>7.3f}{m['pearson']:>9.3f}{m['spearman']:>8.3f}"
              f"{m['auroc']:>7.3f}{m['aurc']:>7.3f}{m['sel'][0.7]:>8.3f}{m['rand']:>7.3f}")
    print("=" * 96)
    print("AUROC>0.5: entropy ranks human-hard items above easy. AURC: lower=better selective risk.")
    print("sel@70: accuracy on the most-confident 70% of items. rand: base accuracy (random select).")

    # ---- r-vs-size curve ----
    curve = [(m["active"], m["total"], name, m["pearson"], m["auroc"])
             for name, m in per_model.items() if m["active"] is not None]
    curve.sort()
    if curve:
        print("\nr-vs-SIZE CURVE (ordered by active params):")
        print(f"  {'active_B':>9}{'total_B':>9}  {'model':<24}{'pearson':>9}{'AUROC':>8}")
        for a, t, name, p, au in curve:
            moe = " *MoE active<total" if a != t else ""
            print(f"  {a:>9.1f}{t:>9.1f}  {name:<24}{p:>9.3f}{au:>8.3f}{moe}")

    # ---- pairwise error-correlation ----
    names = [n for n in per_model if n in err_vecs]
    if len(names) >= 2:
        print("\nPAIRWISE ERROR-CORRELATION (phi on shared items; low = idiosyncratic errors):")
        common = set.intersection(*[set(err_vecs[n].keys()) for n in names])
        common = sorted(common)
        print(f"  shared items across all {len(names)} models: {len(common)}")
        if len(common) >= 10:
            M = np.array([[err_vecs[n][u] for u in common] for n in names])
            print("       " + "".join(f"{n[:8]:>9}" for n in names))
            for i, ni in enumerate(names):
                row = ""
                for j in range(len(names)):
                    if M[i].std() < 1e-9 or M[j].std() < 1e-9:
                        row += f"{'n/a':>9}"
                    else:
                        row += f"{np.corrcoef(M[i], M[j])[0,1]:>9.3f}"
                print(f"  {ni[:6]:<6}{row}")

    out = {n: {k: v for k, v in m.items() if k != "sel"} | {"sel": m["sel"]}
           for n, m in per_model.items()}
    (OUT_DIR / "sweep_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_DIR / 'sweep_eval.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-frontier", action="store_true")
    ap.add_argument("--hard-quantile", type=float, default=0.5,
                    help="H quantile above which an item counts as 'hard' for AUROC")
    ap.add_argument("--prefix", default="sweep_chaosnli_",
                    help="output-file prefix to evaluate; use 'vision_cifar_' for the vision track")
    ap.add_argument("--frontier-file", default="frontier_chaosnli.jsonl",
                    help="frontier results file for --include-frontier (track-specific)")
    args = ap.parse_args()
    evaluate(args.include_frontier, args.hard_quantile, args.prefix, args.frontier_file)


if __name__ == "__main__":
    main()