"""Shared helpers for TRACE experiments: data loading, seeding, statistics.

Reviewer-grade reporting lives here so every experiment uses the same protocol:
multi-seed runs, mean +/- std, 95% CIs, and a non-parametric significance test
(Mann-Whitney U) for every comparison we claim.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Default seed set: 8 seeds (E5 runs 24 sessions). Each experiment defines its own
# SEEDS; this shared default matches the protocol actually run and reported.
SEEDS = tuple(range(8))

# Canonical content-guard hyperparameters, fixed a priori and disclosed in the
# paper.  NOT tuned on any test split.  Shared by every experiment that trains a
# content guard so the configuration is identical and reproducible everywhere.
CONTENT_CFG = dict(n_clauses=200, threshold=25, s=5.0,
                   max_features=256, use_bigrams=True, min_df=2, epochs=25)


def load_dataset(name: str) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """Load a cached dataset -> {'train': (texts, labels), 'test': (...)}.

    Falls back to a deterministic train/test split if only one split exists.
    """
    base = DATA / name
    out: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for split in ("train", "test"):
        p = base / f"{split}.csv"
        if not p.exists():
            continue
        texts, labels = [], []
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                texts.append(row["text"])
                labels.append(int(row["label"]))
        out[split] = (texts, np.array(labels, dtype=np.int64))
    if "train" not in out:
        raise FileNotFoundError(f"no cached data for '{name}' (run scripts/fetch_data.py)")
    if "test" not in out:                      # deterministic 80/20 split
        texts, labels = out["train"]
        rng = np.random.default_rng(12345)
        idx = rng.permutation(len(texts))
        cut = int(0.8 * len(texts))
        tr, te = idx[:cut], idx[cut:]
        out = {"train": ([texts[i] for i in tr], labels[tr]),
               "test": ([texts[i] for i in te], labels[te])}
    return out


#
# statistics
#
def summarize(values: Sequence[float]) -> dict:
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean()) if n else float("nan")
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n,
            "ci95_lo": mean - 1.96 * sem, "ci95_hi": mean + 1.96 * sem,
            "min": float(a.min()) if n else float("nan"),
            "max": float(a.max()) if n else float("nan")}


def mannwhitney(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided Mann-Whitney U with a rank-biserial effect size."""
    from scipy.stats import mannwhitneyu
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0 or (np.ptp(a) == 0 and np.ptp(b) == 0):
        # Both samples are zero-variance constants: a rank test across seeds is
        # meaningless (no sampling variability), so any tiny p-value would be a
        # pure artifact of repeating a deterministic computation.  Report it as
        # degenerate; the deterministic gap is already visible in the means.
        return {"U": float("nan"), "p_value": 1.0, "effect_rank_biserial": 0.0,
                "note": "degenerate (no sampling variability; deterministic gap)"}
    try:
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        rbc = 1.0 - 2.0 * U / (len(a) * len(b))   # rank-biserial correlation
        return {"U": float(U), "p_value": float(p), "effect_rank_biserial": float(rbc)}
    except ValueError as e:
        return {"U": float("nan"), "p_value": 1.0, "effect_rank_biserial": 0.0, "note": str(e)}


def prf1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def write_results(name: str, payload: dict) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return path


def host_info() -> dict:
    import platform
    return {"python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "platform": platform.platform()}
