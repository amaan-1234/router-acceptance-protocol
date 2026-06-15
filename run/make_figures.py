"""Generate the two paper figures from the capacity sweep. No cluster/GPU needed.

Figure 1 (capacity curve): reads outputs/sweep_eval.json and plots, against active
parameter count (log x): Pearson r and AUROC on the left, AURC on a twin axis. The
frontier (Sonnet) is shown as a horizontal reference band since it has no small-model
size. The 4B outlier is annotated.

Figure 2 (error-correlation heatmap): the 6x6 phi matrix between model error vectors on
the shared 450 items. These values are not stored in sweep_eval.json (the matrix is
printed, not serialized), so they are pasted here from the eval_sweep.py run. If you
re-run the sweep, update ERR_MATRIX / ERR_LABELS from the printed matrix.

Usage:
    python -m run.make_figures          # writes outputs/fig_capacity_curve.png
                                        #        outputs/fig_error_correlation.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from cac import config
    OUT = Path(config.OUTPUTS_DIR)
except Exception:
    OUT = Path("outputs")

# ----- size labels for nicer display; keys match sweep_eval.json model names -----
PRETTY = {
    "qwen35_0p8b_sub450": "0.8B",
    "qwen35_2b_sub450": "2B",
    "qwen35_4b_sub450": "4B",
    "qwen35_9b_sub450": "9B",
    "qwen35_27b_fp8_sub450": "27B",
    "qwen35_122b_a10b_fp8_sub450": "122B\n(10B act, MoE)",
    "frontier_sonnet": "Sonnet",
}

# ----- error-correlation matrix from the eval_sweep.py printout (6 small models) -----
# order as printed: 0.8B, 122B, 27B, 2B, 4B, 9B
ERR_LABELS = ["0.8B", "122B(MoE)", "27B", "2B", "4B", "9B"]
ERR_MATRIX = np.array([
    [ 1.000,  -0.050,  0.779,  0.815,  0.314,  0.540],
    [-0.050,   1.000,  0.098,  0.027,  0.177,  0.140],
    [ 0.779,   0.098,  1.000,  0.828,  0.373,  0.588],
    [ 0.815,   0.027,  0.828,  1.000,  0.404,  0.590],
    [ 0.314,   0.177,  0.373,  0.404,  1.000,  0.468],
    [ 0.540,   0.140,  0.588,  0.590,  0.468,  1.000],
])


def _load():
    p = OUT / "sweep_eval.json"
    if not p.exists():
        raise SystemExit(f"missing {p}; run eval_sweep first")
    return json.loads(p.read_text())


def fig_capacity_curve(data):
    # collect sized points (skip frontier, which has no small-model size)
    # x-position uses TOTAL params for intuitive ordering; MoE flagged in label
    pts = []
    for name, m in data.items():
        if name == "frontier_sonnet":
            continue
        tot = m.get("total")
        act = m.get("active")
        if tot is None:
            continue
        is_moe = (act is not None and act != tot)
        pts.append((tot, name, m["pearson"], m["auroc"], m["aurc"], is_moe))
    pts.sort()
    if not pts:
        raise SystemExit("no sized points in sweep_eval.json")

    xs = [p[0] for p in pts]
    labels = [PRETTY.get(p[1], p[1]).split("\n")[0] + ("*" if p[5] else "") for p in pts]
    pear = [p[2] for p in pts]
    auroc = [p[3] for p in pts]
    aurc = [p[4] for p in pts]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.set_xscale("log")
    ax1.plot(xs, pear, "o-", color="#1f4e79", label="Pearson r (entropy vs H)", lw=2, ms=7)
    ax1.plot(xs, auroc, "s--", color="#2e7d32", label="AUROC (ranks hard items)", lw=2, ms=6)
    ax1.set_xlabel("Total parameters (B, log scale)   *MoE: 122B total / 10B active")
    ax1.set_ylabel("Pearson r  /  AUROC")
    ax1.set_ylim(0.0, 0.8)
    ax1.grid(True, which="both", ls=":", alpha=0.4)

    # frontier reference bands
    fr = data.get("frontier_sonnet")
    if fr:
        ax1.axhline(fr["pearson"], color="#1f4e79", alpha=0.35, ls="-", lw=1)
        ax1.axhline(fr["auroc"], color="#2e7d32", alpha=0.35, ls="--", lw=1)
        ax1.text(xs[-1], fr["pearson"] + 0.01, "Sonnet r", color="#1f4e79", fontsize=8, ha="right")
        ax1.text(xs[-1], fr["auroc"] + 0.01, "Sonnet AUROC", color="#2e7d32", fontsize=8, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(xs, aurc, "^:", color="#b00020", label="AURC (lower=better)", lw=2, ms=6)
    ax2.set_ylabel("AURC (selective risk)", color="#b00020")
    ax2.tick_params(axis="y", labelcolor="#b00020")
    ax2.set_ylim(0.0, 0.55)

    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.minorticks_off()

    # annotate 4B outlier
    for x, lab, p in zip(xs, labels, pear):
        if lab == "4B":
            ax1.annotate("4B outlier", xy=(x, p), xytext=(x*0.55, p + 0.12),
                         fontsize=8, color="#444",
                         arrowprops=dict(arrowstyle="->", color="#888", lw=1))

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper left", fontsize=8, framealpha=0.9)
    ax1.set_title("Tracking human disagreement vs. model capacity (ChaosNLI, N=450)", fontsize=10)
    fig.tight_layout()
    out = OUT / "fig_capacity_curve.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def fig_error_correlation():
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(ERR_MATRIX, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ERR_LABELS)))
    ax.set_yticks(range(len(ERR_LABELS)))
    ax.set_xticklabels(ERR_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ERR_LABELS, fontsize=8)
    for i in range(len(ERR_LABELS)):
        for j in range(len(ERR_LABELS)):
            v = ERR_MATRIX[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.55 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("error-vector correlation (phi)", fontsize=8)
    ax.set_title("Pairwise error correlation on shared items\n(low = idiosyncratic; MoE is the outlier)", fontsize=9)
    fig.tight_layout()
    out = OUT / "fig_error_correlation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    data = _load()
    fig_capacity_curve(data)
    fig_error_correlation()


if __name__ == "__main__":
    main()
