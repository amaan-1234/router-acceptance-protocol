"""4B-outlier diagnostic figure for the professor (point 1) on the dense-only axis
(point 4: the 122B MoE is shown as a separate marker, not on the dense capacity line).

Two panels:
  (a) AURC vs dense capacity (lower=better) -- the headline metric per the prof.
      A LOWESS-free simple trend (log-linear fit on the dense points EXCLUDING 4B) is
      drawn so the reader sees the underlying improvement-with-scale trend and how far
      4B sits off it. The MoE and Sonnet are reference markers.
  (b) the residual of each dense point from that trend, so "how much of an outlier is
      4B" is a single bar height.

Reads nothing; uses the final seven-point numbers. Regenerate from sweep_eval.json if
they change.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs") if Path("outputs").exists() else Path(".")

# total params (x), AURC, Pearson; dense models only for the trend
DENSE = [
    ("0.8B", 0.8, 0.465, 0.138),
    ("2B",   2.0, 0.424, 0.132),
    ("4B",   4.0, 0.230, 0.377),
    ("9B",   9.0, 0.306, 0.304),
    ("27B",  27.0, 0.438, 0.271),
]
MOE = ("122B-MoE", 122.0, 0.387, 0.268)   # plotted separately
SONNET = ("Sonnet", 0.178, 0.471)         # reference band

names = [d[0] for d in DENSE]
x = np.array([d[1] for d in DENSE])
aurc = np.array([d[2] for d in DENSE])
pear = np.array([d[3] for d in DENSE])
logx = np.log10(x)

# fit trend on dense points EXCLUDING 4B, so 4B's deviation is measured against the rest
mask = np.array([n != "4B" for n in names])
# AURC trend (expect downward / improving with scale)
ca = np.polyfit(logx[mask], aurc[mask], 1)
aurc_fit = np.polyval(ca, logx)
aurc_resid = aurc - aurc_fit

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

# ---- panel (a): AURC vs capacity with trend ----
xs_smooth = np.linspace(logx.min(), logx.max(), 100)
ax1.plot(10**xs_smooth, np.polyval(ca, xs_smooth), "-", color="#999",
         lw=1.5, label="trend (dense, excl. 4B)")
ax1.plot(x, aurc, "o", color="#1f4e79", ms=9, label="dense models", zorder=5)
# highlight 4B
i4 = names.index("4B")
ax1.plot(x[i4], aurc[i4], "o", color="#d81b60", ms=13, zorder=6, label="4B (outlier)")
# MoE + Sonnet markers
ax1.plot(MOE[1], MOE[2], "D", color="#8e24aa", ms=9, label="122B MoE (separate)")
ax1.axhline(SONNET[1], color="#2e7d32", ls="--", alpha=0.6, lw=1.2)
ax1.text(x.max(), SONNET[1]-0.012, "Sonnet (frontier)", color="#2e7d32", fontsize=8, ha="right")
for n, xv, yv, _ in DENSE:
    ax1.annotate(n, (xv, yv), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax1.set_xscale("log")
ax1.set_xlabel("Total parameters (B, log scale)")
ax1.set_ylabel("AURC (selective risk, lower=better)")
ax1.set_title("(a) Disagreement-tracking improves with scale;\n4B sits well below the trend")
ax1.legend(fontsize=7.5, loc="upper right")
ax1.grid(True, which="both", ls=":", alpha=0.35)

# ---- panel (b): residual-from-trend bars ----
colors = ["#d81b60" if n == "4B" else "#1f4e79" for n in names]
ax2.bar(range(len(names)), aurc_resid, color=colors)
ax2.axhline(0, color="#444", lw=1)
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels(names)
ax2.set_ylabel("AURC residual from trend\n(negative = better than expected)")
ax2.set_title("(b) How much of an outlier is 4B?\nresidual vs the dense trend")
for i, v in enumerate(aurc_resid):
    ax2.text(i, v + (0.006 if v >= 0 else -0.014), f"{v:+.3f}", ha="center", fontsize=8,
             color="#d81b60" if names[i] == "4B" else "#333")
ax2.grid(True, axis="y", ls=":", alpha=0.35)

fig.suptitle("4B outlier diagnostic (ChaosNLI, dense models, N=450)", fontsize=11, y=1.02)
fig.tight_layout()
out = OUT / "fig_4b_outlier.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
print(f"4B AURC residual from dense trend: {aurc_resid[i4]:+.3f} "
      f"(next-largest |residual|: {sorted(abs(aurc_resid))[-2]:.3f})")
