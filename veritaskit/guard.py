"""TRACE content guard: an interpretable TM detector for unsafe text.

The content guard is the *input/output* channel of TRACE.  It classifies a
piece of text (a user prompt, a retrieved document, a tool argument, a model
response) as benign or unsafe (e.g. a prompt-injection / jailbreak attempt),
using a Tsetlin Machine over token-presence literals.

Two properties distinguish it from a neural classifier:

* **Interpretable.**  ``explain()`` lists the learned clauses as readable
  conjunctions of tokens.  A reviewer can see *why* a prompt was flagged.
* **Certifiable.**  ``audit(text)`` returns, for the firing clauses, the exact
  literal-inclusion pattern that fired, ready to be packaged into a signed CNF
  receipt by :mod:`trace.ledger`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .booleanize import BooleanVocabulizer
from .tm import TsetlinMachine, TMConfig, to_literals


@dataclass
class ContentGuardConfig:
    n_clauses: int = 200
    threshold: int = 25
    s: float = 5.0
    n_states_per_action: int = 100
    max_features: int = 256
    use_bigrams: bool = True
    min_df: int = 2
    epochs: int = 30
    seed: int | None = 0


@dataclass
class ContentVerdict:
    unsafe: bool
    margin: int                 # signed vote sum; >0 => unsafe
    fired_clause_ids: List[int] # positive-polarity clauses that fired
    label: int                  # 1 unsafe, 0 benign


class ContentGuard:
    """Interpretable Tsetlin content guard (binary: unsafe=1, benign=0)."""

    def __init__(self, cfg: ContentGuardConfig | None = None):
        self.cfg = cfg or ContentGuardConfig()
        self.vec = BooleanVocabulizer(
            max_features=self.cfg.max_features,
            use_bigrams=self.cfg.use_bigrams,
            min_df=self.cfg.min_df,
        )
        self.tm: TsetlinMachine | None = None

    # training
    def fit(self, texts: Sequence[str], labels: Sequence[int],
            verbose: bool = False) -> "ContentGuard":
        X = self.vec.fit_transform(texts, labels)
        tm_cfg = TMConfig(
            n_clauses=self.cfg.n_clauses,
            n_features=self.vec.n_features,
            n_states_per_action=self.cfg.n_states_per_action,
            threshold=self.cfg.threshold,
            s=self.cfg.s,
            seed=self.cfg.seed,
        )
        self.tm = TsetlinMachine(tm_cfg)
        self.tm.fit(X, np.asarray(labels, dtype=np.int8),
                    epochs=self.cfg.epochs, verbose=verbose)
        return self

    # inference
    def _features(self, text: str) -> np.ndarray:
        return self.vec.transform([text])[0]

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        assert self.tm is not None, "fit() first"
        return self.tm.predict(self.vec.transform(texts))

    def score(self, texts: Sequence[str], labels: Sequence[int]) -> float:
        return float((self.predict(texts) == np.asarray(labels)).mean())

    def verdict(self, text: str) -> ContentVerdict:
        """Full per-sample verdict with the firing positive-polarity clauses."""
        assert self.tm is not None, "fit() first"
        lits = to_literals(self.vec.transform([text]))[0]
        v, co = self.tm.vote_sum(lits, predict=True)
        pos = np.where(self.tm.polarity == 1)[0]
        fired = [int(c) for c in pos if co[c] == 1]
        return ContentVerdict(unsafe=v > 0, margin=int(v),
                              fired_clause_ids=fired, label=1 if v > 0 else 0)

    def audit(self, text: str) -> dict:
        """Return the data a CNF receipt needs: literal vector + per-clause
        include masks for the positive-polarity clauses that fired."""
        assert self.tm is not None, "fit() first"
        lits = to_literals(self.vec.transform([text]))[0]
        co = self.tm.bank.clause_outputs(lits, predict=True)
        inc = self.tm.bank.actions()
        pos = np.where(self.tm.polarity == 1)[0]
        fired = {int(c): inc[c].copy() for c in pos if co[c] == 1}
        return {"literals": lits.astype(np.int8), "fired": fired,
                "n_literals": int(lits.shape[0])}

    # interpretability
    def explain(self, top_k: int = 10, max_literals: int = 8) -> List[str]:
        """Human-readable rendering of the most specific positive-polarity
        clauses (those that vote 'unsafe').  Each is an AND of tokens."""
        assert self.tm is not None, "fit() first"
        inc = self.tm.clause_includes()
        pos = np.where(self.tm.polarity == 1)[0]
        rules = []
        for c in pos:
            lit_idx = np.where(inc[c])[0]
            if 0 < len(lit_idx) <= max_literals:
                terms = [self.vec.feature_name(int(i)) for i in lit_idx]
                rules.append((len(lit_idx), "IF " + " AND ".join(terms) + " -> UNSAFE"))
        # Prefer the most specific (longest) human-readable clauses.
        rules.sort(key=lambda r: r[0], reverse=True)
        return [r[1] for r in rules[:top_k]]
