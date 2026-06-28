"""Exact Boolean encoding of a trained Tsetlin content guard.

The whole premise of VERITAS is that an *interpretable* guard is also a
*verifiable* one: because the guard's decision is a propositional formula, we can
hand it to an exact reasoner and ask questions about **all** inputs at once,
rather than spot-checking finitely many.

A binary Tsetlin Machine guard decides

    vote(x)  = sum_j  polarity_j * clause_j(x)
    UNSAFE(x) <=> vote(x) > 0

where each ``clause_j`` is a conjunction (AND) of *included literals*, a literal
being a feature ``b_i`` ("token i present") or its negation.  In ``predict`` mode
an empty clause (no included literals) outputs 0, exactly the convention the
deployed guard uses (see :mod:`veritaskit.automata`), so the encoding is a
*faithful* model of the guard as it actually runs, not an approximation of it.

This module extracts that formula from a fitted :class:`~veritaskit.guard.ContentGuard`
and emits it in two interchangeable exact forms:

* a **z3** expression (the vote as an integer term), used by the SMT backend; and
* a **CNF + pseudo-Boolean** description, used by the SAT backend (pysat),

plus a pure-NumPy ``eval`` that reproduces the guard's decision bit-for-bit, the
oracle a counterexample is checked against *without trusting any solver*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class GuardEncoding:
    """A solver-agnostic, exact propositional model of a trained guard.

    Attributes, n_features : int
        Number of Boolean features ``F`` (the guard's input dimension).
    clauses : list of (polarity, terms)
        One entry per *non-empty* clause: ``polarity`` in {+1, -1} and ``terms`` a
        list of ``(feature_index, sign)`` with ``sign`` +1 for a positive literal
        ``b_i`` and -1 for a negated literal ``~b_i``.  Empty clauses output 0 in
        predict mode and are dropped (they never contribute to the vote).
    vocab : list of str
        Feature index -> token name (so properties can speak about real tokens).
    """

    n_features: int
    clauses: List[Tuple[int, List[Tuple[int, int]]]]
    vocab: List[str]

    #
    # construction
    #
    @classmethod
    def from_guard(cls, guard) -> "GuardEncoding":
        """Extract the exact decision formula from a fitted ContentGuard."""
        if guard.tm is None:
            raise ValueError("guard must be fit() before it can be encoded")
        inc = np.asarray(guard.tm.clause_includes(), dtype=bool)   # (C, 2F)
        polarity = np.asarray(guard.tm.polarity, dtype=np.int64)   # (C,)
        F = int(guard.vec.n_features)
        clauses: List[Tuple[int, List[Tuple[int, int]]]] = []
        for j in range(inc.shape[0]):
            lits = np.where(inc[j])[0]
            if lits.size == 0:
                continue                       # empty clause -> 0 in predict mode
            terms = [(int(l), 1) if l < F else (int(l) - F, -1) for l in lits]
            clauses.append((int(polarity[j]), terms))
        return cls(n_features=F, clauses=clauses, vocab=list(guard.vec.vocab_))

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)

    def token_feature(self, token: str) -> Optional[int]:
        try:
            return self.vocab.index(token)
        except ValueError:
            return None

    def feature_token(self, i: int) -> str:
        return self.vocab[i] if 0 <= i < len(self.vocab) else f"feat{i}"

    #
    # ground-truth evaluation (the model-free oracle)
    #
    def clause_value(self, j: int, x: np.ndarray) -> int:
        polarity, terms = self.clauses[j]
        if not terms:
            return 0                       # empty clause never fires (predict mode)
        for (i, sign) in terms:
            bit = bool(x[i]) if sign == 1 else (not bool(x[i]))
            if not bit:
                return 0
        return 1

    def vote(self, x: np.ndarray) -> int:
        """Exact integer vote on a Boolean feature vector ``x`` (length F)."""
        x = np.asarray(x).astype(bool)
        v = 0
        for j, (polarity, terms) in enumerate(self.clauses):
            if not terms:
                continue                   # empty clause never fires -> contributes 0
            ok = True
            for (i, sign) in terms:
                bit = x[i] if sign == 1 else (not x[i])
                if not bit:
                    ok = False
                    break
            if ok:
                v += polarity
        return int(v)

    def unsafe(self, x: np.ndarray) -> bool:
        return self.vote(x) > 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorised UNSAFE/SAFE prediction over a feature matrix (n, F)."""
        X = np.asarray(X).astype(bool)
        return np.array([1 if self.vote(X[i]) > 0 else 0 for i in range(X.shape[0])],
                        dtype=np.int8)

    @property
    def n_negative(self) -> int:
        return sum(1 for (p, _) in self.clauses if p < 0)

    def with_must_flag(self, feats) -> "GuardEncoding":
        """Return a *repaired* copy that provably satisfies ``must_flag(t)`` for
        every feature in ``feats``.

        Constructive repair: append ``k = n_negative + 1`` positive single-literal
        clauses ``(+1, [(t, +1)])`` for each token ``t``.  Whenever ``t`` is
        present these contribute ``+k``; the worst case original vote is
        ``-n_negative``; so the repaired vote is ``>= -n_negative + (n_negative+1)
        = 1 > 0``.  The guarantee is by construction and re-verifiable, something
        no amount of retraining can promise."""
        k = self.n_negative + 1
        extra = []
        for t in feats:
            extra += [(1, [(int(t), 1)]) for _ in range(k)]
        return GuardEncoding(self.n_features, list(self.clauses) + extra, list(self.vocab))

    #
    # z3 (SMT) form
    #
    def z3_vote(self, b):
        """Build the integer vote term over an externally-provided list ``b`` of
        F z3 Bools.  Returns ``(vote, unsafe)``.  Sharing ``b`` across two
        encodings is what lets the verifier compare two guards on the *same*
        input (equivalence) or one guard on two related inputs (monotonicity)."""
        import z3

        def lit(i: int, sign: int):
            return b[i] if sign == 1 else z3.Not(b[i])

        terms = []
        for (polarity, cl_terms) in self.clauses:
            conj = z3.And(*[lit(i, s) for (i, s) in cl_terms]) if cl_terms else z3.BoolVal(False)
            terms.append(z3.If(conj, z3.IntVal(polarity), z3.IntVal(0)))
        vote = z3.Sum(terms) if terms else z3.IntVal(0)
        return vote, vote > 0

    def z3_build(self, suffix: str = ""):
        """Return ``(b, vote, unsafe)`` z3 terms over fresh feature Bools.

        ``b`` is a list of F z3 Bools; ``vote`` is an Int term equal to the guard
        vote; ``unsafe`` is the Bool ``vote > 0``.  ``suffix`` lets callers build
        two independent copies of the guard (for monotonicity / equivalence).
        """
        import z3
        b = [z3.Bool(f"b{suffix}_{i}") for i in range(self.n_features)]
        vote, unsafe = self.z3_vote(b)
        return b, vote, unsafe

    def realizability_z3(self, b):
        """Optional constraints so a counterexample corresponds to *plausible
        text*: a bigram feature ``a_b`` can only be present if both unigram
        features ``a`` and ``b`` are.  Returns a list of z3 implications over the
        features that are bigrams whose constituents are also features."""
        import z3
        cons = []
        idx = {t: i for i, t in enumerate(self.vocab)}
        for bg, j in idx.items():
            if "_" in bg:
                a, _, c = bg.partition("_")
                ia, ic = idx.get(a), idx.get(c)
                if ia is not None:
                    cons.append(z3.Implies(b[j], b[ia]))
                if ic is not None:
                    cons.append(z3.Implies(b[j], b[ic]))
        return cons

    #
    # CNF + pseudo-Boolean form (for the SAT backend)
    #
    def to_cnf(self, top: int = 0):
        """Tseitin-encode the clause outputs into CNF.

        Returns ``(cnf, feat_vars, clause_vars, weights, next_var)`` where:
        * ``feat_vars[i]``  is the CNF variable for feature ``b_i`` (1-based),
        * ``clause_vars[j]`` is the variable that is TRUE iff clause j fires,
        * ``weights[j]``    is the clause polarity (+1/-1) for the PB threshold,
        * ``cnf``           is a list of clauses (lists of signed ints) asserting
                            ``clause_var <-> AND(literals)`` for every clause,
        * ``next_var``      is the next free variable id.

        The decision ``UNSAFE`` is ``sum_j weights[j]*clause_vars[j] >= 1``; the
        caller adds that pseudo-Boolean constraint with pysat's ``PBEnc``.
        """
        v = 1
        feat_vars = {}
        for i in range(self.n_features):
            feat_vars[i] = v
            v += 1

        def lit_var(i: int, sign: int) -> int:
            return feat_vars[i] if sign == 1 else -feat_vars[i]

        cnf: List[List[int]] = []
        clause_vars: List[int] = []
        weights: List[int] = []
        for (polarity, terms) in self.clauses:
            cj = v
            v += 1
            clause_vars.append(cj)
            weights.append(polarity)
            lits = [lit_var(i, s) for (i, s) in terms]
            if not lits:
                cnf.append([-cj])              # empty clause never fires (output 0)
                continue
            # cj <-> AND(lits):  (cj -> each lit) and (AND lits -> cj)
            for L in lits:
                cnf.append([-cj, L])           # cj -> L
            cnf.append([cj] + [-L for L in lits])  # (all L) -> cj
        return cnf, feat_vars, clause_vars, weights, v
