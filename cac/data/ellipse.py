"""ELLIPSE corpus loader for the grading-uncertainty track (Phase 2, real data).

Analogue of cac/data/mohler.py. ELLIPSE: ~8,890 ELL argumentative essays (grades
8-12), each scored by TWO trained raters across an Overall holistic score plus
six analytic dimensions (Cohesion, Syntax, Vocabulary, Phraseology, Grammar,
Conventions), each 1.0-5.0 in 0.5 increments. Source: scrosseye/ELLIPSE-Corpus
raw-rater-scores file (password-protected per that repo's README).

NOTE: despite earlier framing, the raw file provides exactly 2 raters per essay
(Rater_1, Rater_2) -- the same rater count as Mohler. The added value over Mohler
is: (a) larger N (8890 vs 2273), (b) full-essay inputs vs short CS answers, and
(c) a 6-dimensional analytic disagreement profile per rater, not just one score.

Human-disagreement target options (selectable downstream):
  - rater_gap    : |Overall_1 - Overall_2|             (0..4, step 0.5; matches Mohler)
  - mean_dim_gap : mean over 6 dims of |dim_1 - dim_2| (finer-grained spread)
  - max_dim_gap  : max over 6 dims of |dim_1 - dim_2|
We default to rater_gap for direct comparability with Mohler.

Place the CSV at data/ellipse/ellipsis_raw_rater_scores_anon_all_essay.csv
(extract from the password-protected zip in scrosseye/ELLIPSE-Corpus; see
fetch() for steps -- the zip password is not redistributed here).
Columns used: text_id_kaggle, Text, Overall_1, Overall_2,
  {Cohesion,Syntax,Vocabulary,Phraseology,Grammar,Conventions}_{1,2},
  Identifying_Info_{1,2}
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from cac import config
    _DATA_DIR = Path(config.DATA_DIR) / "ellipse"
except Exception:
    _DATA_DIR = Path("data/ellipse")

CSV_NAME = "ellipsis_raw_rater_scores_anon_all_essay.csv"
DIMS = ["Cohesion", "Syntax", "Vocabulary", "Phraseology", "Grammar", "Conventions"]
SCORE_MIN, SCORE_MAX = 1.0, 5.0


@dataclass
class EllipseItem:
    uid: str                  # text_id_kaggle
    text: str
    overall_1: float
    overall_2: float
    dims_1: dict[str, float]  # {Cohesion: x, ...} rater 1
    dims_2: dict[str, float]  # rater 2
    identifying_info: int     # 1 if either rater flagged PII in the text

    @property
    def rater_gap(self) -> float:
        return abs(self.overall_1 - self.overall_2)

    @property
    def mean_dim_gap(self) -> float:
        return float(np.mean([abs(self.dims_1[d] - self.dims_2[d]) for d in DIMS]))

    @property
    def max_dim_gap(self) -> float:
        return float(np.max([abs(self.dims_1[d] - self.dims_2[d]) for d in DIMS]))

    @property
    def human_grade(self) -> float:
        """Adjudicated overall grade = mean of the two raters (the routing 'truth')."""
        return (self.overall_1 + self.overall_2) / 2.0


def fetch(dest: Path | None = None) -> Path:
    """Locate the ELLIPSE raw-rater CSV; raises with instructions if absent.

    Unlike mohler.fetch(), this cannot auto-download: the source zip on
    github.com/scrosseye/ELLIPSE-Corpus is password-protected (password
    'ellipse_raw_data', per that repo's README). Extract it once manually:

        git clone https://github.com/scrosseye/ELLIPSE-Corpus.git /tmp/ellipse_src
        unzip -P ellipse_raw_data \
            /tmp/ellipse_src/ellipsis_raw_rater_scores_anon_all_essay.zip \
            -d <DATA_DIR>/ellipse/
    """
    dest = dest or (_DATA_DIR / CSV_NAME)
    if not dest.exists():
        raise FileNotFoundError(
            f"missing {dest}. See cac.data.ellipse.fetch.__doc__ for extraction steps.")
    return dest


def load(path: Path | None = None, drop_identifying: bool = False) -> list[EllipseItem]:
    path = path or (_DATA_DIR / CSV_NAME)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; see cac.data.ellipse.fetch()")
    items: list[EllipseItem] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                o1 = float(r["Overall_1"]); o2 = float(r["Overall_2"])
                d1 = {d: float(r[f"{d}_1"]) for d in DIMS}
                d2 = {d: float(r[f"{d}_2"]) for d in DIMS}
                ident = int(r["Identifying_Info_1"]) or int(r["Identifying_Info_2"])
            except (TypeError, ValueError, KeyError):
                continue
            if drop_identifying and ident:
                continue
            items.append(EllipseItem(
                uid=r["text_id_kaggle"], text=r["Text"],
                overall_1=o1, overall_2=o2, dims_1=d1, dims_2=d2,
                identifying_info=ident))
    return items


def human_disagreement(items: list[EllipseItem], kind: str = "rater_gap") -> np.ndarray:
    """(N,) per-item human grader disagreement -- the correlation/routing target."""
    if kind == "rater_gap":
        return np.array([it.rater_gap for it in items], dtype=np.float64)
    if kind == "mean_dim_gap":
        return np.array([it.mean_dim_gap for it in items], dtype=np.float64)
    if kind == "max_dim_gap":
        return np.array([it.max_dim_gap for it in items], dtype=np.float64)
    raise ValueError(f"unknown kind {kind}")


def human_grades(items: list[EllipseItem]) -> np.ndarray:
    """(N,) adjudicated overall grade (rater mean) -- the accuracy target for routing."""
    return np.array([it.human_grade for it in items], dtype=np.float64)


def stratified_subset(items: list[EllipseItem], n: int = 450,
                       kind: str = "rater_gap", seed: int = 0) -> list[EllipseItem]:
    """n-item subset stratified by human disagreement, oversampling the
    nonzero-gap tail -- matches the ChaosNLI/CIFAR-10H convention, since
    rater_gap is heavily zero-skewed (median == 0)."""
    gap = human_disagreement(items, kind)
    rng = np.random.default_rng(seed)
    zero = [i for i, g in enumerate(gap) if g == 0]
    nonzero = [i for i, g in enumerate(gap) if g > 0]
    n_nonzero = min(len(nonzero), n // 2)
    n_zero = min(n - n_nonzero, len(zero))
    idx = list(rng.choice(nonzero, size=n_nonzero, replace=False)) + \
          list(rng.choice(zero, size=n_zero, replace=False))
    rng.shuffle(idx)
    return [items[i] for i in idx]


if __name__ == "__main__":
    p = fetch()
    items = load(p)
    gap = human_disagreement(items)
    dimgap = human_disagreement(items, "mean_dim_gap")
    print(f"loaded {len(items)} items")
    print(f"rater_gap (Overall):   mean={gap.mean():.3f} max={gap.max():.1f} "
          f"frac_disagree(>0)={np.mean(gap > 0):.3f}")
    print(f"mean_dim_gap (6 dims): mean={dimgap.mean():.3f} max={dimgap.max():.3f}")
    print(f"mean adjudicated overall grade: {human_grades(items).mean():.3f}")
    sub = stratified_subset(items)
    print(f"stratified subset: {len(sub)} items, "
          f"mean rater_gap={human_disagreement(sub).mean():.3f}")
