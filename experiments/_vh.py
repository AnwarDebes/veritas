"""Shared helpers for the VERITAS experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veritaskit import ContentGuard, ContentGuardConfig, GuardEncoding


def train_guard(Xtr, ytr, C=20, F=40, T=12, s=4.0, epochs=40, seed=0,
                use_bigrams=True, min_df=2) -> ContentGuard:
    cfg = ContentGuardConfig(n_clauses=C, threshold=T, s=s, max_features=F,
                             use_bigrams=use_bigrams, min_df=min_df, epochs=epochs, seed=seed)
    return ContentGuard(cfg).fit(Xtr, ytr)


def positive_literal_tokens(enc: GuardEncoding) -> List[str]:
    """Tokens that appear as a *positive* literal in some *unsafe-voting* clause, a natural audited cue lexicon: every such token is a reason the guard fires."""
    toks = set()
    for (polarity, terms) in enc.clauses:
        if polarity > 0:
            for (i, sign) in terms:
                if sign == 1:
                    toks.add(enc.vocab[i])
    return sorted(toks)


def attack_lexicon(enc: GuardEncoding, top: int = 16) -> List[str]:
    """The guard's top discriminative tokens (vocab order) that exist as features, the tokens a deployer would most want a 'must-flag' guarantee for."""
    return [t for t in enc.vocab[:top]]


def attack_exclusive_tokens(guard, Xtr, ytr, min_pos: int = 3, top: int = 8) -> List[str]:
    """Vocab tokens that appear in unsafe training docs but NEVER in benign ones, so ``must_flag(t)`` is a TRUE obligation on the training distribution and a
    learnable/repairable target (sorted by unsafe frequency)."""
    from veritaskit.booleanize import tokenize
    ytr = np.asarray(ytr)
    pos, neg = {}, {}
    for t, y in zip(Xtr, ytr):
        tk = tokenize(t)
        terms = set(tk) | set(f"{a}_{b}" for a, b in zip(tk, tk[1:]))
        tgt = pos if y == 1 else neg
        for term in terms:
            tgt[term] = tgt.get(term, 0) + 1
    cand = [t for t in guard.vec.vocab_ if pos.get(t, 0) >= min_pos and neg.get(t, 0) == 0]
    cand.sort(key=lambda t: pos.get(t, 0), reverse=True)
    return cand[:top]


def fuzz_find_bypass(enc: GuardEncoding, must_present: Sequence[int],
                     n_probes: int, rng: np.random.Generator,
                     density: float = 0.15) -> Tuple[bool, int]:
    """Red-team a 'must_flag' obligation by random probing.

    Generate ``n_probes`` random feature vectors that satisfy ``must_present``
    (those features set as required) and check whether ANY is labelled SAFE (a
    bypass).  Returns ``(found, probes_used)``, ``found`` True if a bypass was
    hit.  This is the stand-in for finite red-teaming: it can only ever sample."""
    F = enc.n_features
    for k in range(1, n_probes + 1):
        x = (rng.random(F) < density).astype(np.int8)
        for (i, val) in must_present:
            x[i] = val
        if not enc.unsafe(x):
            return True, k
    return False, n_probes
