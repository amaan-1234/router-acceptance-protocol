"""Mohler ASAG loader for the grading-uncertainty track (Phase 2, real data).

Analogue of cac/data/chaosnli.py. Mohler 2011 "Texas Extended": CS data-structures
short answers, each scored 0-5 by TWO instructors (score_me, score_other) plus the
average. This is the per-rater human disagreement the project measures model
disagreement against — the grading analogue of CIFAR-10H soft labels / ChaosNLI
label distributions.

Human-disagreement target options (selectable downstream):
  - rater_gap   : |score_me - score_other|        (0..5; the natural 2-rater spread)
  - rater_std   : population std of the 2 scores   (= gap/2; monotone w/ gap)
  - rater_var   : variance                          (= (gap/2)^2)
We default to rater_gap. With only 2 raters this is coarse (6 possible values),
and the dataset is correct-answer biased (mean avg ~4.17) — both noted as limits;
ELLIPSE (>=3 analytic raters) is the higher-resolution follow-up.

Place the CSV at  data/mohler/mohler_dataset_edited.csv  (download URL in fetch()).
Columns: id, question, desired_answer, student_answer, score_me, score_other, score_avg
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from cac import config  # available inside the Phase 1 repo
    _DATA_DIR = Path(config.DATA_DIR) / "mohler"
except Exception:  # standalone / testing
    _DATA_DIR = Path("data/mohler")

CSV_NAME = "mohler_dataset_edited.csv"
CSV_URL = (
    "https://raw.githubusercontent.com/gsasikiran/"
    "Comparative-Evaluation-of-Pretrained-Transfer-Learning-Models-on-ASAG/"
    "master/comparative_evaluation_on_mohler_dataset/dataset/mohler_dataset_edited.csv"
)
SCORE_MAX = 5.0


@dataclass
class MohlerItem:
    uid: str               # the question.answer id, e.g. "1.1"
    question: str
    desired_answer: str
    student_answer: str
    score_me: float
    score_other: float
    score_avg: float

    @property
    def rater_gap(self) -> float:
        return abs(self.score_me - self.score_other)

    @property
    def rater_std(self) -> float:
        return float(np.std([self.score_me, self.score_other]))

    @property
    def human_grade(self) -> float:
        """Adjudicated grade = mean of the two raters (the routing 'truth')."""
        return self.score_avg


def fetch(dest: Path | None = None) -> Path:
    """Download the Mohler CSV if not present. Returns the local path."""
    dest = dest or (_DATA_DIR / CSV_NAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        import urllib.request
        print(f"[mohler] downloading -> {dest}")
        urllib.request.urlretrieve(CSV_URL, dest)
    return dest


def load(path: Path | None = None) -> list[MohlerItem]:
    path = path or (_DATA_DIR / CSV_NAME)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run cac.data.mohler.fetch() or download {CSV_URL}")
    items: list[MohlerItem] = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                me = float(r["score_me"]); oth = float(r["score_other"])
                avg = float(r["score_avg"]) if r.get("score_avg") not in (None, "") \
                    else (me + oth) / 2.0
            except (TypeError, ValueError):
                continue  # skip rows with unparseable scores
            items.append(MohlerItem(
                uid=r["id"], question=r["question"],
                desired_answer=r["desired_answer"], student_answer=r["student_answer"],
                score_me=me, score_other=oth, score_avg=avg))
    return items


def human_disagreement(items: list[MohlerItem], kind: str = "rater_gap") -> np.ndarray:
    """(N,) per-item human grader disagreement — the correlation/routing target."""
    if kind == "rater_gap":
        return np.array([it.rater_gap for it in items], dtype=np.float64)
    if kind == "rater_std":
        return np.array([it.rater_std for it in items], dtype=np.float64)
    if kind == "rater_var":
        return np.array([it.rater_std ** 2 for it in items], dtype=np.float64)
    raise ValueError(f"unknown kind {kind}")


def human_grades(items: list[MohlerItem]) -> np.ndarray:
    """(N,) adjudicated grade (rater mean) — the accuracy target for routing."""
    return np.array([it.human_grade for it in items], dtype=np.float64)


if __name__ == "__main__":
    p = fetch()
    items = load(p)
    gap = human_disagreement(items)
    print(f"loaded {len(items)} items")
    print(f"rater_gap: mean={gap.mean():.3f} max={gap.max():.0f} "
          f"frac_disagree(>0)={np.mean(gap > 0):.3f}")
    print(f"mean adjudicated grade: {human_grades(items).mean():.3f}")
