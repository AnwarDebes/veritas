"""Binary Tsetlin Machine on the ETTA substrate.

A clean reference implementation of the binary Tsetlin Machine (Granmo, 2018)
built on :class:`trace.automata.ETTABank`.  TRACE's content guard is a thin
wrapper around this trainer.  With ``lambda_decay = trace_alpha = 0`` it is a
vanilla TM; the trace machinery is inert.

We keep everything pure NumPy and per-sample so that the include/exclude state
of every clause, and therefore every CNF receipt TRACE later signs, is
exactly the quantity being trained, with no hidden bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .automata import ETTABank


def to_literals(X: np.ndarray) -> np.ndarray:
    """(n_samples, n_features) 0/1 -> (n_samples, 2*n_features) raw+negated."""
    X = np.asarray(X, dtype=np.int8)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    if not np.all((X == 0) | (X == 1)):
        raise ValueError("X must be binary 0/1")
    return np.concatenate([X, 1 - X], axis=1)


@dataclass
class TMConfig:
    n_clauses: int
    n_features: int
    n_states_per_action: int = 100
    threshold: int = 15
    s: float = 3.9
    lambda_decay: float = 0.0
    trace_alpha: float = 0.0
    seed: int | None = None


class TsetlinMachine:
    """Binary Tsetlin Machine with Echo-Trace automata.

    ``n_clauses`` is the TOTAL clause count; the first half are positive
    polarity (vote +1 for class 1), the second half negative.  Must be even.
    """

    def __init__(self, cfg: TMConfig):
        if cfg.n_clauses % 2 != 0:
            raise ValueError("n_clauses must be even (half +polarity, half -polarity).")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.bank = ETTABank(
            n_clauses=cfg.n_clauses,
            n_literals=2 * cfg.n_features,
            n_states_per_action=cfg.n_states_per_action,
            lambda_decay=cfg.lambda_decay,
            trace_alpha=cfg.trace_alpha,
            rng=self.rng,
        )
        self.polarity = np.empty(cfg.n_clauses, dtype=np.int8)
        self.polarity[: cfg.n_clauses // 2] = 1
        self.polarity[cfg.n_clauses // 2:] = -1

    # inference
    def vote_sum(self, literals_row: np.ndarray, predict: bool) -> tuple[int, np.ndarray]:
        co = self.bank.clause_outputs(literals_row, predict=predict)
        v = int((self.polarity.astype(np.int32) * co.astype(np.int32)).sum())
        return v, co

    def decision_margin(self, X: np.ndarray) -> np.ndarray:
        """Signed vote sum per sample (positive -> class 1).  A calibrated-free
        confidence proxy used by the content guard."""
        lits = to_literals(X)
        return np.array([self.vote_sum(lits[i], predict=True)[0]
                         for i in range(lits.shape[0])], dtype=np.int32)

    def predict_one(self, literals_row: np.ndarray) -> int:
        v, _ = self.vote_sum(literals_row, predict=True)
        return 1 if v > 0 else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        lits = to_literals(X)
        return np.array([self.predict_one(lits[i]) for i in range(lits.shape[0])],
                        dtype=np.int8)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())

    # training
    def _train_step(self, literals_row: np.ndarray, y: int) -> None:
        v, co = self.vote_sum(literals_row, predict=False)
        T = self.cfg.threshold
        v_clipped = max(-T, min(T, v))

        # Canonical binary TM (Granmo, 2018): ONE feedback-selection probability
        # per training example is applied to clauses of BOTH polarities; polarity
        # only switches Type I <-> Type II feedback, not the probability.
        if y == 1:
            p = (T - v_clipped) / (2.0 * T)
            self._feedback_subset(co, literals_row, +1, "I", p)
            self._feedback_subset(co, literals_row, -1, "II", p)
        else:
            p = (T + v_clipped) / (2.0 * T)
            self._feedback_subset(co, literals_row, +1, "II", p)
            self._feedback_subset(co, literals_row, -1, "I", p)

    def _feedback_subset(self, clause_outputs, literals, polarity, ftype, p) -> None:
        if p <= 0.0:
            return
        idx = np.where(self.polarity == polarity)[0]
        if idx.size == 0:
            return
        chosen = self.rng.random(idx.size) < p
        mask = np.zeros(self.cfg.n_clauses, dtype=bool)
        mask[idx[chosen]] = True
        if not mask.any():
            return
        if ftype == "I":
            self.bank.type_I_feedback(clause_outputs, literals, s=self.cfg.s,
                                      feedback_mask=mask)
        else:
            self.bank.type_II_feedback(clause_outputs, literals, feedback_mask=mask)

    def fit(self, X, y, epochs: int = 10, shuffle: bool = True, verbose: bool = False):
        lits = to_literals(X)
        y = np.asarray(y, dtype=np.int8)
        for epoch in range(epochs):
            order = np.arange(lits.shape[0])
            if shuffle:
                self.rng.shuffle(order)
            for i in order:
                self._train_step(lits[i], int(y[i]))
            if verbose:
                print(f"[TM] epoch {epoch+1}/{epochs} train_acc={self.score(X, y):.4f}")
        return self

    # interpretability
    def clause_includes(self) -> np.ndarray:
        """(n_clauses, 2*n_features) bool include mask, the audit surface."""
        return self.bank.actions()
