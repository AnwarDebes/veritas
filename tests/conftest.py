import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veritaskit import ContentGuard, ContentGuardConfig, GuardEncoding


def make_corpus(seed=0, n=60):
    rng = np.random.default_rng(seed)
    bad = ["ignore", "override", "disregard"]
    ok = ["summarise", "weather", "please", "report", "ticket", "schedule"]

    def make(k, unsafe):
        rows = []
        for _ in range(k):
            toks = list(rng.choice(ok, size=int(rng.integers(2, 5))))
            if unsafe:
                toks.append(str(rng.choice(bad)))
            rng.shuffle(toks)
            rows.append(" ".join(map(str, toks)))
        return rows

    texts = make(n, True) + make(n, False)
    labels = [1] * n + [0] * n
    return texts, labels, bad, ok


@pytest.fixture(scope="session")
def tiny_guard():
    texts, labels, bad, ok = make_corpus()
    cfg = ContentGuardConfig(n_clauses=20, threshold=8, s=3.0, max_features=12,
                             use_bigrams=False, min_df=1, epochs=30, seed=1)
    g = ContentGuard(cfg).fit(texts, labels)
    return g, bad, ok


@pytest.fixture(scope="session")
def tiny_enc(tiny_guard):
    g, bad, ok = tiny_guard
    return GuardEncoding.from_guard(g), bad, ok
