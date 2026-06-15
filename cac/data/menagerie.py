"""Menagerie loader for the code-grading uncertainty track (Phase 2, real multi-rater data).
import os

Menagerie (Messer et al., SIGCSE 2025): real second-semester CS1 Java submissions
(predator/prey simulator), graded post hoc by groups of 4 assessors on 4 criteria
(Correctness, Code Elegance, Readability, Documentation). 279 submissions x 4 criteria,
EXACTLY 4 graders per (submission, criterion) -- a balanced 4-rater design.

This is the higher-resolution grading analogue of CIFAR-10H/ChaosNLI: where Mohler
had 2 instructors (coarse |gap|), Menagerie has 4 graders, so per-item human
disagreement is a real standard deviation over 4 ordinal scores -- the finest human-
disagreement target in the project, and the one the multi-rater literature recommends.

Grades are letters A++ .. F, mapped to an ordinal 0..13 scale (rank-preserving; the
spread/disagreement is what matters, not the absolute value). 'NAN' rows are dropped.

Data layout (relative to the Menagerie repo root):
  data/grades.csv                                      long: one row per (submission,criterion,grader)
  data/anonymised_assignments/<year>/<sub>/*.java      the Java source per submission

Human-disagreement target per (submission, criterion): std of the 4 grader scores.
Adjudicated grade per (submission, criterion): mean of the 4 grader scores.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from cac import config
    _REPO = Path(config.DATA_DIR).parent  # cac repo root; Menagerie sits beside it
except Exception:
    _REPO = Path(".")

# Default: the cloned Menagerie repo. Override via set_root().
_MENAGERIE_ROOT = Path(os.environ.get("MENAGERIE_ROOT", "data/Menagerie"))

CRITERIA = ["Correctness", "Code Elegance", "Readability", "Documentation"]

# Ordinal letter -> number. Rank-preserving, evenly spaced. F=0 .. A++=13.
GRADE_MAP = {
    "F": 0, "D-": 1, "D": 2, "D+": 3, "C-": 4, "C": 5, "C+": 6,
    "B-": 7, "B": 8, "B+": 9, "A-": 10, "A": 11, "A+": 12, "A++": 13,
}
GRADE_MAX = 13.0


def set_root(path: str | Path) -> None:
    global _MENAGERIE_ROOT
    _MENAGERIE_ROOT = Path(path)


@dataclass
class MenagerieItem:
    submission_id: str            # assignment_number as string
    criterion: str
    grader_scores: list[float]    # the 4 ordinal grades
    java_path: str | None = None  # submission dir (filled lazily for the agent demo)

    @property
    def rater_std(self) -> float:
        return float(np.std(self.grader_scores))

    @property
    def rater_range(self) -> float:
        return float(max(self.grader_scores) - min(self.grader_scores))

    @property
    def adjudicated(self) -> float:
        return float(np.mean(self.grader_scores))


@dataclass
class MenagerieSubmission:
    submission_id: str
    java_files: dict[str, str] = field(default_factory=dict)   # filename -> source
    per_criterion: dict[str, MenagerieItem] = field(default_factory=dict)


def _grades_df(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "data" / "grades.csv")
    df = df[df["grade"] != "NAN"].copy()
    df["score"] = df["grade"].map(GRADE_MAP)
    df = df.dropna(subset=["score"])
    # assignment_number arrives as a float (e.g. 44.0); normalize to a clean int-string
    df["submission_id"] = (
        df["assignment_number"].astype(float).astype(int).astype(str)
    )
    return df


def load_items(root: Path | None = None) -> list[MenagerieItem]:
    """One MenagerieItem per (submission, criterion) with the 4 grader scores."""
    root = Path(root or _MENAGERIE_ROOT)
    if not (root / "data" / "grades.csv").exists():
        raise FileNotFoundError(f"grades.csv not under {root}; clone Menagerie there")
    df = _grades_df(root)
    items: list[MenagerieItem] = []
    for (sub, crit), g in df.groupby(["submission_id", "skill"]):
        scores = [float(s) for s in g["score"].tolist()]
        if len(scores) < 2:            # need >=2 graders to have disagreement
            continue
        items.append(MenagerieItem(submission_id=sub, criterion=crit, grader_scores=scores))
    return items


def human_disagreement(items: list[MenagerieItem], kind: str = "rater_std") -> np.ndarray:
    if kind == "rater_std":
        return np.array([it.rater_std for it in items], dtype=np.float64)
    if kind == "rater_range":
        return np.array([it.rater_range for it in items], dtype=np.float64)
    raise ValueError(kind)


def adjudicated_grades(items: list[MenagerieItem]) -> np.ndarray:
    return np.array([it.adjudicated for it in items], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Java source loading (for the grader ensemble + the agent demo)
# --------------------------------------------------------------------------- #
def load_java_sources(submission_id: str, root: Path | None = None,
                      max_chars: int = 24000) -> dict[str, str]:
    """Return {filename: source} for a submission.

    The Java lives only inside the per-year zips under
    data/anonymised_assignments/<year>.zip, with internal paths shaped like
    '<year>/<year>_Submission_<id>/<File>.java' (e.g. '18~19/18~19_Submission_44/Fox.java').
    We match on the '_Submission_<id>/' segment with an exact-id boundary so that
    submission 4 does not also pick up 44, 144, etc. Truncates to max_chars.
    """
    root = Path(root or _MENAGERIE_ROOT)
    sid = str(submission_id).split(".")[0]          # '44.0' -> '44'
    needle = f"_Submission_{sid}/"
    out: dict[str, str] = {}
    zdir = root / "data" / "anonymised_assignments"
    for z in sorted(zdir.glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist()
                         if needle in n and n.endswith(".java")]
                for n in names:
                    out[Path(n).name] = zf.read(n).decode("utf-8", "replace")
                if names:
                    break                           # found the submission's year
        except zipfile.BadZipFile:
            continue
    # budget-trim: keep whole files until the char cap
    trimmed, total = {}, 0
    for name, src in sorted(out.items()):
        if total + len(src) > max_chars and trimmed:
            break
        trimmed[name] = src
        total += len(src)
    return trimmed


if __name__ == "__main__":
    items = load_items()
    std = human_disagreement(items)
    print(f"loaded {len(items)} (submission x criterion) items, "
          f"{len(set(i.submission_id for i in items))} submissions")
    print(f"rater_std: mean={std.mean():.3f} max={std.max():.3f} "
          f"frac_disagree(std>0)={np.mean(std > 0):.3f}")
    print(f"mean adjudicated grade (0-13): {adjudicated_grades(items).mean():.3f}")
    by_c = {}
    for it in items:
        by_c.setdefault(it.criterion, []).append(it.rater_std)
    for c, v in by_c.items():
        print(f"  {c:<15} n={len(v):<4} mean rater_std={np.mean(v):.3f}")
    # spot-check java loading on the first submission
    sid = items[0].submission_id
    src = load_java_sources(sid)
    print(f"java for submission {sid}: {len(src)} files, "
          f"{sum(len(s) for s in src.values())} chars")
